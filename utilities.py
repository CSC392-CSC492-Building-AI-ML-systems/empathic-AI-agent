import random
import string


def generate_random_session_id():
    # Define the characters to choose from
    characters = string.ascii_letters + string.digits
    # Generate a random session id of length 7
    random_string = ''.join(random.choices(characters, k=7))
    return random_string
