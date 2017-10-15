def Gassuion(img,percetage):
    noise = np.random.random([img.shape[0], img.shape[1]])*255
    t = np.random.random([img.shape[0], img.shape[1]])*percetage
    temp = (1 - t) * img + t * noise
    return  temp
