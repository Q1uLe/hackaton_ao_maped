FROM --platform=linux/amd64 ros:humble-ros-base-jammy
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN apt-get update && apt-get install ros-humble-rmw-cyclonedds-cpp -y
RUN useradd participant
USER participant
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENTRYPOINT ["/entrypoint.sh"]
