import numpy as np
import pandas as pd
import random

args = {
    'lr': 0.01
}

def get_normalization_params(data):
    # para pegar os parâmetros de normalização, pois é necessário normalizar os dados de teste com os mesmos parâmetros dos dados de treino
    return {
        'min': data.min(),
        'max': data.max()
    }

def _normalize(sample, norm_params):
    return (sample - norm_params['min']) / (norm_params['max'] - norm_params['min'])

def pre_processing(df, inputs: list, desired_output='d', train=True, norm_params=None):
    x = df[inputs].to_numpy()
    # se tivermos um conjunto de treino, calculamos os parâmetros de normalização, senão, usamos os parâmetros passados
    if train:
        norm_params = get_normalization_params(x) if norm_params is None else norm_params
        x = np.apply_along_axis(_normalize, 0, x, norm_params)
        d = df[desired_output].to_numpy()
        d = d.reshape(d.shape[0], 1).transpose()
    else:
        if norm_params:
            x = np.apply_along_axis(_normalize, 0, x, norm_params) # normaliza os dados de teste com os mesmos parâmetros dos dados de treino
        else:
            raise ValueError('norm_params must be provided for test data') # se não tivermos os parâmetros de normalização, não podemos normalizar os dados de teste
    
    x = x.transpose() # transpõe para que cada coluna seja uma amostra
    x = np.insert(x, 0, -1 * np.ones(x.shape[1]), axis=0) # adiciona o bias, que é sempre -1 na primeira linha
    return (x, d, norm_params) if train else x

def create_weights(x):
    w = np.ones(x.shape[0]) * [random.random() for _ in range(x.shape[0])]
    return w.reshape(w.shape[0], 1) # reshape para que seja um vetor coluna
    
def train(x, d, w):
    initial = f'Vetor de pesos inicial: {w.ravel().copy()}' # copia para que não seja alterado
    num_epochs = 0
    while True:
        error = False
        for i in range(d.shape[1]):
            u = np.dot(w.T, x[:, i]) # seleciona-se a coluna, representativa de cada amostra
            y_pred = np.sign(u)
            if y_pred[0] != d[:, i][0]:
                w += args['lr'] * (d[:,i] - y_pred[0]) * x[:, i].reshape(-1, 1) # reshape para que o x seja vetor coluna também, para a soma funcionar
                error = True
        num_epochs += 1
        print(f'Época {num_epochs} - Vetor de pesos: {w.ravel()}') # ravel para que seja um vetor linha
        if not error:
            break
    print(initial)
    print(f'Vetor de pesos final: {w.ravel()}')
    return w

def inference(x, w):
    u = w.T @ x # multiplicação matricial
    y_pred = np.sign(u).ravel()

    c1 = list(filter(lambda x: x[1] < 0, enumerate(y_pred))) # filtra os valores cuja classe é C1
    c2 = list(filter(lambda x: x[1] >= 0, enumerate(y_pred))) 
    print('='*50, 'Inferência', '='*50)
    print("Amostras pertencentes à classe C1:", *list(map(lambda x: x[0]+1, c1))) # mapeia as amostras da classe pelos índices
    print(f"Amostras pertencentes à classe C2:", *list(map(lambda x: x[0]+1, c2))) 

if __name__ == '__main__':
    # train
    train_df = pd.read_csv('train.csv')
    x, d, norm_params = pre_processing(train_df, ['x1', 'x2', 'x3'], 'd')
    w = create_weights(x)
    w = train(x, d, w)
    inference(x, w)
    
    # test
    test_df = pd.read_csv('test.csv')
    x = pre_processing(test_df, ['x1', 'x2', 'x3'], train=False, norm_params=norm_params)
    inference(x, w)
