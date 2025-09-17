try:
    import librosa  # Try to import the librosa library
    print("librosa is installed!")  # If successful, print this message
except ImportError:
    print("librosa is NOT installed.")  # If not installed, print this message
