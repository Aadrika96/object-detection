import time
import edgeiq
import requests
import time
from Synth import Synth

synth_room_id = "65c506a6b6bca40b8106cc32"
synth_key = "mr_8884f292683f3f9ad57b66c092d560568a7d39d46cadac293e2f450e893a3fa3"

def main():
    i=0
    obj_detect = edgeiq.ObjectDetection("alwaysai/mobilenet_ssd")
    obj_detect.load(engine=edgeiq.Engine.DNN)
    cap = edgeiq.WebcamVideoStream(cam=0)

    print("Loaded model:\n{}\n".format(obj_detect.model_id))
    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Labels:\n{}\n".format(obj_detect.labels))

    fps = edgeiq.FPS()
    synth = Synth(cap, synth_room_id, synth_key)

    try:
        with cap as video_stream:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            person_count = 0  # Initialize count outside the loop
            while True:
                frame = video_stream.read()

                results = obj_detect.detect_objects(frame, confidence_level=.5)
                frame = edgeiq.markup_image(frame, results.predictions, colors=obj_detect.colors)

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append("Inference time: {:1.3f} s".format(results.duration))
                text.append("Objects:")

                for prediction in results.predictions:
                    if prediction.label == 'person':
                        person_count += 1  # Increment count if "person" detected
                    text.append("{}: {:2.2f}%".format(prediction.label, prediction.confidence * 100))
                
                try:
                    #print("approx. FPS: {:.2f}".format(fps.compute_fps()))
                    streamer.send_data(frame)
                    synth.publish_frame(frame)
                except Exception as e:
                    print(f"Error in publishing : {e}")

                if not synth.is_connected():
                    print("Connection to RTMP server lost. Reattempting connection...")
                    try:
                        time.sleep(10)
                        synth.reconnect()
                        print("Reconnection successful.")
                    except Exception as e:
                        print(f"Reconnection failed: {e}")

                # Publish person count every few frames or as needed
                if person_count > 0:  # Publish count only if there are detected persons
                    # print(f"Number of persons detected: {person_count}")
                    synth.publish_data("occupants", str(person_count))
                    person_count = 0  # Reset count after publishing

                fps.update()
                # time.sleep(3)

    finally:
        fps.stop()
        synth.close()
        # print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        # print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        # print("Program Ending")


if __name__ == "__main__":
    main()
 
