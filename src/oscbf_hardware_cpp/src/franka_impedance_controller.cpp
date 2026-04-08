#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <Eigen/Dense>
#include <franka/robot.h>
#include <franka/model.h>

#include <thread>
#include <atomic>
#include <mutex>
#include <string.h>

using namespace std::chrono_literals;


/**
 * Sets a default collision behavior, joint impedance and Cartesian impedance.
 *
 * @param[in] robot Robot instance to set behavior on.
 */

//Default behavior with lower collision thresholds and higher impedance for more rigid behavior.
void setDefaultBehavior(franka::Robot& robot) {
    robot.setCollisionBehavior(
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}}, {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}}, {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}}, {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}}, {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}});
    robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
    // Cartesian Impedance: {{x, y, z, roll, pitch, yaw}}
    robot.setCartesianImpedance({{3000, 3000, 3000, 300, 300, 300}});
  }

// Default behavior with higher collision thresholds and lower impedance for more compliant behavior.
// void setDefaultBehavior(franka::Robot& robot) {
//     robot.setCollisionBehavior(
//         {{40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0}}, {{40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0}},
//         {{40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0}}, {{40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0}},
//         {{40.0, 40.0, 40.0, 40.0, 40.0, 40.0}}, {{40.0, 40.0, 40.0, 40.0, 40.0, 40.0}},
//         {{40.0, 40.0, 40.0, 40.0, 40.0, 40.0}}, {{40.0, 40.0, 40.0, 40.0, 40.0, 40.0}});

//     robot.setJointImpedance({{300.0, 300.0, 300.0, 250.0, 250.0, 200.0, 200.0}});
//     // robot.setJointImpedance({{0, 0, 0, 0, 0, 0, 0}});
//     // {{x, y, z, roll, pitch, yaw}}
//     robot.setCartesianImpedance({{300.0, 300.0, 300.0, 30.0, 30.0, 30.0}});
//     // robot.setCartesianImpedance({{0, 0, 0, 0, 0, 0}});
// }
  
/**
 * @class FrankaImpedanceController
 * @brief A ROS2 node for controlling a Franka Panda robot using impedance control
 * 
 * This class provides a ROS2 interface for communicating with and controlling 
 * a Franka Panda robotic arm. It supports:
 * - Connecting to the robot via network
 * - Publishing joint states
 * - Receiving and applying torque commands
 * - Running a real-time control loop
 * 
 * Key features:
 * - Subscribes to torque commands
 * - Publishes current joint states at 1000 Hz
 * - Manages robot connection and control in a separate thread
 * - Handles potential exceptions during robot communication
 */
class FrankaImpedanceController : public rclcpp::Node {
public:
    /**
     * @brief Construct a new Franka Impedance Controller
     * 
     * Initializes ROS2 subscribers, publishers, and establishes 
     * connection with the Franka robot
     */
    FrankaImpedanceController() : Node("franka_impedance_controller"), stop_control_loop_(false) {
        RCLCPP_INFO(this->get_logger(), "Starting Controller...");

        // Declare parameters. Use default robot hostname if not provided
        this->declare_parameter<std::string>("robot_hostname", "172.16.0.2");
        this->declare_parameter<bool>("realtime", false);

        // Create QoS
        rclcpp::QoS qos(rclcpp::KeepLast(10));
        qos.best_effort();
        qos.durability_volatile();


        // Torque subscription: Receives torque commands from ROS2 topic
        torque_subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "franka/torque_command", qos,
            std::bind(&FrankaImpedanceController::torqueCallback, this, std::placeholders::_1)
        );

        // Joint state publisher: Broadcasts current robot joint states
        joint_state_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "franka/joint_states", qos
        );

        // Create a timer for publishing joint states at 1000 Hz
        joint_state_timer_ = this->create_wall_timer(
            1ms, 
            std::bind(&FrankaImpedanceController::publishJointStates, this)
        );

        // Initialize robot connection
        std::string robot_hostname = this->get_parameter("robot_hostname").as_string();
        try {
            if (this->get_parameter("realtime").as_bool()) {
                robot_ = std::make_unique<franka::Robot>(robot_hostname, franka::RealtimeConfig::kEnforce);
            } else {
                robot_ = std::make_unique<franka::Robot>(robot_hostname, franka::RealtimeConfig::kIgnore);
            }
            RCLCPP_INFO(this->get_logger(), "Robot connected: %s", robot_hostname.c_str());
            // setDefaultBehavior(*robot_);
            // model_ = std::make_unique<franka::Model>(robot_->loadModel());
            // RCLCPP_INFO(this->get_logger(), "Model loaded.");
        } catch (const std::exception& ex) {
            RCLCPP_ERROR(this->get_logger(), "Failed to connect to robot: %s", ex.what());
        }

        // Initialize last robot state
        if (robot_) {
            last_robot_state_ = robot_->readOnce();
        }

        // Start control loop in a separate thread
        try {
            control_thread_ = std::thread(&FrankaImpedanceController::controlLoop, this);
        } catch (const std::exception& ex) {
            RCLCPP_ERROR(this->get_logger(), "Failed to start control loop: %s", ex.what());
        }
    }

    /**
     * @brief Destructor to safely shutdown the robot control loop
     * 
     * Signals the control loop to stop and waits for the thread to join
     */
    ~FrankaImpedanceController() {
        stop_control_loop_ = true; // Signal the control loop to stop
        if (control_thread_.joinable()) {
            control_thread_.join(); // Wait for it to finish
        }
    }

private:
    /**
     * @brief Publishes current joint states of the robot
     * 
     * Reads the current robot state and publishes joint positions 
     * and velocities to the ROS2 topic
     */
    void publishJointStates() {
        if (!robot_) {
            RCLCPP_WARN(this->get_logger(), "Robot not initialized");
            return;
        }

        try {
            // Create a local copy of the last robot state
            franka::RobotState current_state;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                current_state = last_robot_state_;
            }

            // Prepare joint state message
            sensor_msgs::msg::JointState joint_state_msg;
            joint_state_msg.header.stamp = this->now();
            joint_state_msg.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};
            
            // Copy joint positions and velocities
            joint_state_msg.position.assign(last_robot_state_.q.begin(), last_robot_state_.q.end());
            joint_state_msg.velocity.assign(last_robot_state_.dq.begin(), last_robot_state_.dq.end());

            // Publish joint states
            joint_state_publisher_->publish(joint_state_msg);

        } catch (const std::exception& ex) {
            RCLCPP_ERROR(this->get_logger(), "Joint state update error: %s", ex.what());
        }
    }

    /**
     * @brief Callback for receiving torque commands
     * 
     * @param msg ROS2 message containing torque commands
     * 
     * Receives torque commands from a ROS2 topic and stores them 
     * for use in the control loop. Ensures thread-safe access 
     * to torque commands.
     */
    void torqueCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        if (!robot_) {
            RCLCPP_WARN(this->get_logger(), "Robot not initialized");
            return;
        }

        if (msg->data.size() < 7) {
            RCLCPP_ERROR(this->get_logger(), "Received torque command with insufficient data");
            return;
        }

        std::lock_guard<std::mutex> lock(torque_mutex_);
        std::copy(msg->data.begin(), msg->data.begin() + 7, tau_command_.begin());
    }

    /**
     * @brief Main control loop for the robot
     * 
     * Runs in a separate thread and continuously applies 
     * received torque commands to the robot. Supports 
     * graceful shutdown via stop_control_loop_ flag.
     */
    void controlLoop() {
        if (!robot_) return;

        RCLCPP_INFO(this->get_logger(), "Starting impedance control loop...");
        
        try {
            robot_->control([this](const franka::RobotState& state, franka::Duration) -> franka::Torques {
                if (stop_control_loop_) {
                    RCLCPP_INFO(this->get_logger(), "Stopping control loop...");
                    franka::Torques zero_torque = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
                    return franka::MotionFinished(zero_torque);
                }

                // Store the latest robot state
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    last_robot_state_ = state;
                }

                std::array<double, 7> tau;
                {
                    std::lock_guard<std::mutex> lock(torque_mutex_);
                    tau = tau_command_; // Use latest received torque command
                }
                
                return tau;
            });

        } catch (const std::exception& ex) {
            RCLCPP_ERROR(this->get_logger(), "Control loop error: %s", ex.what());
        }
    }

    // Robot and model pointers
    std::unique_ptr<franka::Robot> robot_;
    // std::unique_ptr<franka::Model> model_;

    // Last known robot state
    franka::RobotState last_robot_state_;

    // ROS publishers, subscriptions, and timer
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr torque_subscription_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
    rclcpp::TimerBase::SharedPtr joint_state_timer_;

    // Control thread and synchronization
    std::thread control_thread_;
    std::atomic<bool> stop_control_loop_;
    std::mutex torque_mutex_;
    std::mutex state_mutex_;
    std::array<double, 7> tau_command_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // Default zero torques
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FrankaImpedanceController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
