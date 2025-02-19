import numpy as np
import pandas as pd
import random

args = {
    'lr': 0.01
}

def pre_processing(df, inputs: list, desired_output='d', train=True):
    x = df[inputs].to_numpy()
    if train:
        d = df[desired_output].to_numpy()
        d = d.reshape(d.shape[0], 1).transpose()
    x = x.transpose()
    x = np.insert(x, 0, -1 * np.ones(x.shape[1]), axis=0)
    return (x, d) if train else x

def create_weights(x):
    w = np.ones(x.shape[0]) * [random.random() for _ in range(x.shape[0])]
    return w.reshape(w.shape[0], 1)
    
def train(x, d, w):
    weight_results = []
    initial = f'Vetor de pesos inicial: {w.ravel().copy()}'
    num_epochs = 0
    while True:
        error = False
        for i in range(d.shape[1]):
            u = np.dot(w.T, x[:, i]) # seleciona-se a coluna, representativa de cada amostra
            y_pred = np.sign(u)
            if y_pred[0] != d[:, i][0]:
                w += args['lr'] * (d[:,i] - y_pred[0]) * x[:, i].reshape(-1, 1) # reshape para que o x seja vetor coluna também, para a soma funcionar
                error = True
        weight_results.append(w.ravel().copy())    
        num_epochs += 1
        print(f'Época {num_epochs} - Vetor de pesos: {w.ravel()}')
        if not error:
            break
    print(initial)
    print(f'Vetor de pesos final: {w.ravel()}')
    return w, weight_results

def inference(x, w):
    u = w.T @ x # multiplicação matricial
    y_pred = np.sign(u).ravel()

    c1 = list(filter(lambda x: x[1] < 0, enumerate(y_pred)))
    c2 = list(filter(lambda x: x[1] >= 0, enumerate(y_pred)))
    print('='*50, 'Inferência', '='*50)
    print("Amostras pertencentes à classe C1:", *list(map(lambda x: x[0]+1, c1)))
    print(f"Amostras pertencentes à classe C2:", *list(map(lambda x: x[0]+1, c2)))

if __name__ == '__main__':
    # train
    train_df = pd.read_csv('train.csv')
    x, d = pre_processing(train_df, ['x1', 'x2', 'x3'], 'd')
    w = create_weights(x)
    w, _ = train(x, d, w)
    inference(x, w)
    
    # test
    test_df = pd.read_csv('test.csv')
    x = pre_processing(test_df, ['x1', 'x2', 'x3'], train=False)
    inference(x, w)
