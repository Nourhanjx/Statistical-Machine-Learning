def evaluate(net, images, labels):
    total_acc = 0    
    loss = 0
    batch_size = 1
    len_layers= net.lay_num
    for batch_index in range(0, images.shape[0], batch_size):
        x = images[batch_index]
        y_target = labels[batch_index]
        for layer in range(len_layers):
            #doing a forward pass for the x value with layers. 
            output = net.layers[layer].forward(x) 
        #calculating the loss
        loss = loss + cross_entropy(output, y_target)
        #predicted class
        if np.argmax(output) == np.argmax(y_target):
            print("accuracy increased!!!")
            total_acc += 1
    accuracy = total_acc/images.shape[0]
    loss=loss/images.shape[0]
    return accuracy,loss
