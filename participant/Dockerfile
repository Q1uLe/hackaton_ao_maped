FROM --platform=linux/amd64 ros:humble-ros-base-jammy
SHELL ["/bin/bash", "-c"]
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN apt-get update && apt-get install ros-humble-rmw-cyclonedds-cpp -y
RUN apt-get install python3-pip python3-colcon-common-extensions -y
RUN pip3 install --no-cache-dir numpy mujoco==3.3.7 scipy
RUN apt-get install -y ros-humble-rosidl-generator-dds-idl
WORKDIR /opt/participant
COPY ros2_ws/src/ ros2_ws/src/
COPY models/ models
COPY t3/go1_t3/ t3/go1_t3/
COPY t3/weights/ t3/weights/
COPY t3/licenses/ t3/licenses/
COPY t3/submission.json t3/submission.json
WORKDIR /opt/participant/ros2_ws
RUN source /opt/ros/humble/setup.bash && colcon build
RUN useradd -m participant
USER participant
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENTRYPOINT ["/entrypoint.sh"]
