from neural import Neural

if __name__ == "__main__":
    neural = Neural()
    neural.train()

    '''test_images, test_labels = MNIST("./train_data/").load_testing()

    total = len(test_images)
    valid = 0
    invalid = []

    for i in range(0, total):
        img = test_images[i]
        predicted = neural.analyze_mnist(img)
        true = test_labels[i]
        if predicted == true:
            valid = valid + 1
        else:
            invalid.append({"image": img, "predicted": predicted, "true": true})

    print("accuracy {}".format(valid / total * 100))'''
    print(neural.analyze("samples/on.png"))
    print(neural.analyze("samples/one.png"))
    print(neural.analyze("samples/seven-3.jpg"))
    print(neural.analyze("samples/eleve.jpg"))
    print(neural.analyze("samples/four.Png"))
