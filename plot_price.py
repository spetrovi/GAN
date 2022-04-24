import matplotlib.pyplot as plt


def show_preds(name):

    f = open(name, 'r')
    string = f.read()
    f.close()

    real = string.split('\n')[0].split(',')[:-1]
    pred = string.split('\n')[1].split(',')[:-1]

    real = list(map(lambda x: float(x), real))
    pred = list(map(lambda x: float(x), pred))


    plt.plot(real, label='lows_real')
    plt.plot(pred, label='lows_predicted')
    plt.legend()
    plt.grid()
    plt.show()
    #plt.savefig('lows.svg', dpi=1000)
    #We should compute the mean error for last X days so we can know how much leisure we have

def show_loss(path, line_name):

    f = open(path, 'r')
    string = f.read()
    f.close()

    real = string.split('\n')[0].split(',')[:-1]


    real = list(map(lambda x: float(x), real))


    plt.plot(real, label=line_name)

    plt.legend()
#    plt.show()
    plt.savefig(path + '.svg', dpi=100)
    plt.close()
    #We should compute the mean error for last X days so we can know how much leisure we have

#show_loss('losses/a_loss', 'a_loss')
#show_loss('losses/d_loss', 'd_loss')







