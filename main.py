import numpy as np
import pandas as pd
import random

args = {
    'lr': 0.01
}

def _get_normalization_params(data):
    # para pegar os parâmetros de normalização, pois é necessário normalizar os dados de teste com os mesmos parâmetros dos dados de treino
    return {
        'min': data.min(),
        'max': data.max()
    }

def _normalize(sample, norm_params):
    return (sample - norm_params['min']) / (norm_params['max'] - norm_params['min'])

def pre_processing(df, in_features: list, desired_output='d', train=True, norm_params=None):
    inputs = df[in_features].to_numpy()
    # se tivermos um conjunto de treino, calculamos os parâmetros de normalização, senão, usamos os parâmetros passados
    if train:
        norm_params = _get_normalization_params(inputs) if norm_params is None else norm_params
        inputs = np.apply_along_axis(_normalize, 0, inputs, norm_params)
        desired_output = df[desired_output].to_numpy()
        desired_output = desired_output.reshape(desired_output.shape[0], 1).transpose()
    else:
        if norm_params:
            inputs = np.apply_along_axis(_normalize, 0, inputs, norm_params) # normaliza os dados de teste com os mesmos parâmetros dos dados de treino
        else:
            raise ValueError('norm_params must be provided for test data') # se não tivermos os parâmetros de normalização, não podemos normalizar os dados de teste
    
    inputs = inputs.transpose() # transpõe para que cada coluna seja uma amostra
    inputs = np.insert(inputs, 0, -1 * np.ones(inputs.shape[1]), axis=0) # adiciona o bias, que é sempre -1 na primeira linha
    return (inputs, desired_output, norm_params) if train else inputs

def create_weights(inputs):
    weights = np.ones(inputs.shape[0]) * [random.random() for _ in range(inputs.shape[0])]
    return weights.reshape(inputs.shape[0], 1) # reshape para que seja um vetor coluna
    
def train(inputs, desired_output, weights):
    initial = f'Vetor de pesos inicial: {weights.ravel().copy()}' # copia para que não seja alterado
    num_epochs = 0
    while True:
        error = False
        for i in range(desired_output.shape[1]):
            activation_potential = np.dot(weights.T, inputs[:, i]) # seleciona-se a coluna, representativa de cada amostra
            y_pred = np.sign(activation_potential)
            if y_pred[0] != desired_output[:, i][0]:
                weights += args['lr'] * (desired_output[:,i] - y_pred[0]) * inputs[:, i].reshape(-1, 1) # reshape para que o x seja vetor coluna também, para a soma funcionar
                error = True
        num_epochs += 1
        print(f'Época {num_epochs} - Vetor de pesos: {weights.ravel()}') # ravel para que seja um vetor linha
        if not error:
            break
    print(initial)
    print(f'Vetor de pesos final: {weights.ravel()}')
    return weights

def inference(inputs, weights):
    activation_potential = weights.T @ inputs # multiplicação matricial
    y_pred = np.sign(activation_potential).ravel()

    c1 = list(filter(lambda inputs: inputs[1] < 0, enumerate(y_pred))) # filtramos as amostras pertencentes à classe C1, através dos índices
    c2 = list(filter(lambda inputs: inputs[1] >= 0, enumerate(y_pred))) # o mesmo da linha acima, para C2
    print('='*50, 'Inferência', '='*50)
    print("Amostras pertencentes à classe C1:", *list(map(lambda inputs: inputs[0]+1, c1))) # mapeia as amostras da classe
    print(f"Amostras pertencentes à classe C2:", *list(map(lambda inputs: inputs[0]+1, c2))) 

if __name__ == '__main__':
    # train
    train_df = pd.read_csv('train.csv')
    inputs, desired_outputs, norm_params = pre_processing(train_df, ['x1', 'x2', 'x3'], 'd')
    weights = create_weights(inputs)
    weights = train(inputs, desired_outputs, weights)
    inference(inputs, weights)
    
    # test
    test_df = pd.read_csv('test.csv')
    inputs = pre_processing(test_df, ['x1', 'x2', 'x3'], train=False, norm_params=norm_params)
    inference(inputs, weights)
