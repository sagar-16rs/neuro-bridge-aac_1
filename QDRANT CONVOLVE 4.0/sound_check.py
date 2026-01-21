import pyttsx3
engine = pyttsx3.init()
engine.say("Testing audio system. One, two, three.")
engine.runAndWait()
print("Did you hear that?")