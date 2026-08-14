import os
import unittest
import wave
from pathlib import Path

os.environ["ACESTEP_ENGINE"] = "contract"

from predict import Predictor, first_audio_part


class AceStepCppCogTests(unittest.TestCase):
    def test_contract_prediction_is_weight_free(self):
        predictor = Predictor()
        predictor.setup()
        output = Path(predictor.predict("soft piano", duration=10, bpm=90, key="D minor", seed=3, steps=8))
        self.assertTrue(output.is_file())
        with wave.open(str(output)) as handle:
            self.assertEqual(handle.getframerate(), 16000)
            self.assertGreater(handle.getnframes(), 1000)

    def test_prompt_and_bpm_are_validated(self):
        predictor = Predictor()
        predictor.setup()
        with self.assertRaises(ValueError):
            predictor.predict("", duration=10)
        with self.assertRaises(ValueError):
            predictor.predict("soft piano", duration=10, bpm=12)

    def test_audio_is_selected_from_native_multipart_result(self):
        content_type = 'multipart/mixed; boundary="native-result"'
        body = (
            b"--native-result\r\nContent-Type: audio/mpeg\r\n\r\nID3music\r\n"
            b"--native-result\r\nContent-Type: application/octet-stream\r\n\r\nlatent\r\n"
            b"--native-result--\r\n"
        )
        data, detected = first_audio_part(body, content_type)
        self.assertEqual(data, b"ID3music")
        self.assertEqual(detected, "audio/mpeg")


if __name__ == "__main__":
    unittest.main()
