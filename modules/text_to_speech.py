import pyttsx3

def speak(text: str):
    """
    Convert text to speech using pyttsx3 (offline TTS).
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)       
    engine.setProperty('volume', 1.0)    

    print("🗣️ Speaking...")
    engine.say(text)
    engine.runAndWait()
