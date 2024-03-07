ARG ALWAYSAI_HW="default"
FROM alwaysai/edgeiq:${ALWAYSAI_HW}-2.2.1
RUN sudo apt update && \
    sudo apt install -y ffmpeg

