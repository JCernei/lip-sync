from skimage import io
import face_alignment


class LandmarksExtractor:
    def __init__(self, device="cuda", landmarks_type="2D", flip=False):
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D
            if landmarks_type == "2D"
            else face_alignment.LandmarksType.THREE_D,
            flip_input=flip,
            device=device,
            face_detector="sfd",
        )

        self.landmarks = []

    def cuda(self):
        return self

    def extract_landmarks(self, image):
        # image: either a path to an image or a numpy array (H, W, C) or tensor batch  (B, C, H, W)
        if isinstance(image, str):
            image = io.imread(image)
        if len(image.shape) == 3:
            preds = self.fa.get_landmarks(image)
        else:
            preds = self.fa.get_landmarks_from_batch(image)

        return preds
