from torchvision import transforms

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

def process_state(state):
    state = preprocess(state)  # shape: (1, 84, 84)
    return state.view(-1)      # flatten to (84*84,)
