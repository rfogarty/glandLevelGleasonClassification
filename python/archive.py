
import math
import pickle

def saveHistory(filename,H) :
    with open(filename, 'wb') as file_pi:
            pickle.dump(H.history, file_pi)
    
    history_dict = H.history
    print(history_dict.keys())


