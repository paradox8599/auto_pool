from pool import Pool
import keyboard as kb

if __name__ == "__main__":
    while kb.read_key() != "shift":
        pass
    Pool().run()
