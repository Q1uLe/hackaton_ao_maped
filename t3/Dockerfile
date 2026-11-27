FROM ros:humble-ros-base-jammy AS ros-runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl python3-pip ros-humble-rmw-cyclonedds-cpp ros-humble-rosidl-generator-dds-idl \
    && rm -rf /var/lib/apt/lists/*

FROM ros-runtime AS interface-build
ARG UNITREE_ROS2_REF=668d1ec5a05d1c38d3306bdca7d59f2ba3581a88
WORKDIR /opt/unitree
RUN mkdir -p src/unitree_go \
    && curl -fsSL --retry 3 "https://codeload.github.com/unitreerobotics/unitree_ros2/tar.gz/${UNITREE_ROS2_REF}" \
        | tar xz -C src/unitree_go --strip-components=5 --wildcards '*/cyclonedds_ws/src/unitree/unitree_go/*' \
    && source /opt/ros/humble/setup.bash \
    && colcon build --merge-install --packages-select unitree_go --cmake-args -DBUILD_TESTING=OFF

FROM ros-runtime
RUN python3 -m pip install --no-cache-dir numpy==1.26.4 mujoco==3.3.7 \
    && useradd --uid 10001 --create-home --shell /bin/bash participant
COPY --from=interface-build /opt/unitree/install /opt/unitree/install
WORKDIR /opt/go1-t3
COPY go1_t3 go1_t3
COPY assets assets
COPY weights weights
COPY licenses licenses
COPY submission.json .
COPY --chmod=755 entrypoint.sh /entrypoint.sh
ENV MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER 10001:10001
ENTRYPOINT ["/entrypoint.sh"]
