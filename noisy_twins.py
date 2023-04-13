# Pseudo Code for StyleGAN generator training with NoisyTwins
# styleG: styleGAN generator
# styleD: styleGAN discriminator
# class_embedding: class embedding layer
# optimizer: optimizer for the generator
# batch_size: overall batch size for training
# latent_dim: latent dimension of the generator
# per_class_sigma: noise augmentation standard deviation for the class embeddings
# lambda_noisy: weight for the noisy twins loss
# num_classes: number of classes in the dataset


def train_StyleGAN_generator(styleG, class_embedding, optimizer, batch_size, 
                            ,latent_dim, per_class_sigma, lambda_noisy, num_classes):
    # Train the generator
    styleG.train()
    optimizer.zero_grad()
    # Generate a batch of latent vectors
    zs = torch.randn(batch_size//2, latent_dim)

    # Generate a batch of class embeddings
    class_labels = torch.randint(0, num_classes, (batch_size//2,))
    class_embeddings = class_embedding(class_labels)


    # Get two batches of augmented class embeddings
    class_embeddings_batch_a = class_embeddings + torch.randn_like(class_embeddings) * per_class_sigma[class_labels] 
    class_embeddings_batch_b = class_embeddings + torch.randn_like(class_embeddings) * per_class_sigma[class_labels] 

    # get output of w vectors from mapping network keeping same zs
    aug_ws_batch_a = styleG.mapping(zs, class_embeddings_batch_a)
    aug_ws_batch_b = styleG.mapping(zs, class_embeddings_batch_b)

    # calculate the noisy twins loss based on the barlow twins for the two augmented batches
    # (https://github.com/facebookresearch/barlowtwins/blob/8e8d284ca0bc02f88b92328e53f9b901e86b4a3c/main.py#L187)
    noisy_twins_loss = calculate_noisy_twins_loss(aug_ws_batch_a, aug_ws_batch_b)

    # merge the ws vectors from both augmented batches to generate images
    ws = torch.cat((aug_ws_batch_a, aug_ws_batch_b), dim=0)
    # get output from synthesis network
    imgs = styleG.synthesis(ws)

    # Calculate the adversarial loss for the generator
    adv_loss = adversarial_loss(styleD(imgs, class_labels))
    
    # Calculate the total loss
    loss = adv_loss + lambda_noisy * noisy_twins_loss
    # Backpropagate the loss

    loss.backward()
    optimizer.step()
    return loss