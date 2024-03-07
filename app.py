import time
import edgeiq
import requests
import time
from Synth import Synth

synth_room_id = "65c79dbd9863ec99529bd7c0"
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
        with cap as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                frame = video_stream.read()
                results = obj_detect.detect_objects(frame, confidence_level=.5)
                frame = edgeiq.markup_image(
                        frame, results.predictions, colors=obj_detect.colors)

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                text.append("Objects:")

                for prediction in results.predictions:
                    if prediction.label == 'person':
                        i = i+1
                    text.append("{}: {:2.2f}%".format(
                        prediction.label, prediction.confidence * 100))

                streamer.send_data(frame, text)
                try:
                    synth.publish_frame(frame)
                except Exception as e:
                    print(f"Error in publishing : {e}")

                print(i)
                y = str(i)
                # api_url= 'https://api.thingspeak.com/update?api_key=ZERGKYOT2LEJ1QOK&field1='
                # api_url = api_url+y
                # response=requests.get(api_url)
                synth.publish_data("occupants", y)

                # if response.status_code== 200:
                #     print('Data Sent...')
                i=0
                time.sleep(16)
                fps.update()
            
                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        synth.close()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
 
