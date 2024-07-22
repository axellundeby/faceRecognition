from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import cv2

PERSON_OF_INTEREST = 'path to person you want to detect'
GROUP_PHOTO = 'path for group photo'
OUTPUT_PHOTO = 'outputImage.jpeg'
THRESHOLD = 0.75
SHOW_NOT_EQUAL = True
is_person_of_interest_in_image = False

def load_image(image_path):
    img = Image.open(image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    return img


def draw_faces(image, box, distance, threshold=THRESHOLD):
    global is_person_of_interest_in_image 
    top_left = (int(box[0]), int(box[1]))
    bottom_right = (int(box[2]), int(box[3]))

    if distance <= threshold:
        color = (0,255,0)
        is_person_of_interest_in_image = True
    elif (SHOW_NOT_EQUAL): 
        color = (255,0,0)
    else: return image

    cv2.rectangle(image, top_left, bottom_right, color, 5)
 
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f'{distance:.2f}'
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = top_left[0] + (bottom_right[0] - top_left[0] - text_size[0]) // 2
    text_y = bottom_right[1] + text_size[1] + 5
    cv2.putText(image, text, (text_x, text_y), font, 1, color, 2, cv2.LINE_AA)
    return image

def compute_person_of_interest_embedding(mtcnn, resnet, device):
    poi_image = load_image(PERSON_OF_INTEREST)
    poi_cropped = mtcnn(poi_image)
    if poi_cropped is not None:
        poi_cropped = poi_cropped.unsqueeze(0) if poi_cropped.ndim == 3 else poi_cropped
        poi_embedding = resnet(poi_cropped.to(device)).detach().cpu()
        return poi_embedding
    else:
        print("No face detected in the person of interest image.")
        return None

def compute_group_faces_embeddings(mtcnn, resnet, device, poi_embedding):
    group_image = load_image(GROUP_PHOTO)
    group_faces = mtcnn(group_image)
    embedding_for_group = []
    if group_faces is not None:
        for face in group_faces:
            face = face.unsqueeze(0) if face.ndim == 3 else face
            embedding = resnet(face.to(device)).detach().cpu()
            distance = torch.linalg.norm(poi_embedding - embedding).item()
            embedding_for_group.append(distance)
    else:
        print("No faces detected in the group photo.")
        return None
    return group_image, group_faces, embedding_for_group

def annotate_image_with_faces(faces, distanceList):
    image = cv2.imread(GROUP_PHOTO)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for face, distance in zip(faces, distanceList):
        if face is not None:
            image = draw_faces(image, face, distance)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUTPUT_PHOTO, image)
    print(f"Annotated image saved to {OUTPUT_PHOTO}")

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device, keep_all=True
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    person_of_intrest_embedding = compute_person_of_interest_embedding(mtcnn, resnet, device)
    if person_of_intrest_embedding is None:
        return

    group_image, group_faces, distanceList = compute_group_faces_embeddings(mtcnn, resnet, device, person_of_intrest_embedding)
    if group_faces is None:
        return



    faces, _ = mtcnn.detect(group_image)
    if faces is None:
        print("No faces detected in the group photo.")
        return
    annotate_image_with_faces(faces, distanceList) 

if __name__ == '__main__':
    main()
    print(is_person_of_interest_in_image)
  
