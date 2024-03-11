from time import sleep
from edgeiq import ObjectDetection, Engine, markup_image
from Synth import Synth
from cv2 import VideoCapture

synth_room_id = "65d86696c0076e9e522d2882"
synth_key = "mr_8884f292683f3f9ad57b66c092d560568a7d39d46cadac293e2f450e893a3fa3"

person_count = 0  # Initialize count outside the loop
prev_person_count = 0

obj_detect = ObjectDetection("alwaysai/mobilenet_ssd")
obj_detect.load(engine=Engine.DNN)

cap = VideoCapture(0)
synth = Synth(cap, synth_room_id, synth_key)

while cap.isOpened():
    while True:
        ret, frame = cap.read()

        if not ret:
            print("frame read failed")
            break

        results = obj_detect.detect_objects(frame, confidence_level=.5)
        frame = markup_image(frame, results.predictions, colors=obj_detect.colors)

        for prediction in results.predictions:
            if prediction.label == 'person':
                person_count += 1  # Increment count if "person" detected

        # Publish data only if person_count changes
        if person_count != prev_person_count:
            try:
                synth.publish_data("occupants", str(person_count))
            except Exception as e:
                print(f"Error in publishing : {e}")
            prev_person_count = person_count

        try:
            synth.publish_frame(frame)
        except Exception as e:
            print(f"Error in publishing : {e}")

        if not synth.is_connected():
            print("Connection to RTMP server lost. Reattempting connection...")
            try:
                sleep(10)
                synth.reconnect()
                print("Reconnection successful.")
            except Exception as e:
                print(f"Reconnection failed: {e}")

        person_count = 0  # Reset count after publishing

cap.release()
synth.close()
