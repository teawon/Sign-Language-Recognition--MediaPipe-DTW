import pandas as pd
import numpy as np
from collections import Counter

from utils.dtw import dtw_distances
from models.sign_model import SignModel
from utils.landmark_utils import extract_landmarks


class SignRecorder(object):
    def __init__(self, reference_signs: pd.DataFrame, seq_len=100):
        # 초기 모드 값 설정 (데이터를 모으는 길이, recording여부 설정)
        self.is_recording = False
        self.seq_len = seq_len

        # List of results stored each frame
        self.recorded_results = []

        # DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        self.reference_signs = reference_signs

    def record(self):
        # recording시작 - 시작 시점부터 데이터를 모은다

        self.reference_signs["distance"].values[:] = 0
        self.is_recording = True

    def process_results(self, results) -> (str, bool):
       

        if self.is_recording:
            # 1. 데이터가 덜 쌓였다면 랜드마크값을 더 추가
    
            if len(self.recorded_results) < self.seq_len:
                self.recorded_results.append(results)
            else:
            # 2. 데이터가 충분하다면 계산 시작
                self.compute_distances()

        if np.sum(self.reference_signs["distance"].values) == 0:
            return "", self.is_recording
        return self._get_sign_predicted(), self.is_recording

    def compute_distances(self):
        """
        Updates the distance column of the reference_signs
        and resets recording variables
        """
        left_hand_list, right_hand_list = [], []
        for results in self.recorded_results:
            _, left_hand, right_hand = extract_landmarks(results)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)

        print("왼쪽 좌표 추출")
        print(left_hand_list)
        print("오른쪽 좌표 추출")
        print(right_hand_list)

        # Create a SignModel object with the landmarks gathered during recording
        recorded_sign = SignModel(left_hand_list, right_hand_list)

        # Compute sign similarity with DTW (ascending order)
        self.reference_signs = dtw_distances(recorded_sign, self.reference_signs)

        # Reset variables
        self.recorded_results = []
        self.is_recording = False

    def _get_sign_predicted(self, batch_size=5, threshold=0.5):
        """
        Method that outputs the sign that appears the most in the list of closest
        reference signs, only if its proportion within the batch is greater than the threshold

        :param batch_size: Size of the batch of reference signs that will be compared to the recorded sign
        :param threshold: If the proportion of the most represented sign in the batch is greater than threshold,
                        we output the sign_name
                          If not,
                        we output "Sign not found"
        :return: The name of the predicted sign
        """
        # Get the list (of size batch_size) of the most similar reference signs
        sign_names = self.reference_signs.iloc[:batch_size]["name"].values
        print("sign_name")
        print(sign_names)

        # Count the occurrences of each sign and sort them by descending order
        sign_counter = Counter(sign_names).most_common()
        print("sign_counter")
        print(sign_counter)


        predicted_sign, count = sign_counter[0]
        # if count / batch_size < threshold:
        #     return "Signe inconnu"
        return predicted_sign
