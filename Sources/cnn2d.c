#include "m_pd.h" // Importa as funções prontas do Pure Data
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"


/* Esse código é uma implementação de uma rede neural auto encoder convolucional.
O objeto cria arrays de matrizes para cada camada da rede (matriz de entrada, matriz de kernel, matriz de convolução e matriz de pooling, etc). 
Somente as matrizes de kernel são inicializadas com valores. As matrizes de entrada são preenchidas com zeros.
Cada matriz de cada camada é liberada e alocada conforme necessário (redimensionamento)
Cada matriz de cada camada possui dimensões específicas que são calculadas 
a partir do nº de camadas, dimensões dos dados de entrada, dimensões dos kernels, pooling, padding e stride.
Os nº de dimensões de cada matriz, stride e janela de pooling são armazenadas em arrays de vetores de floats (input_padded, convolv_matriz_size, pooling_matriz_size, kernels_size, pooling_size, stride_conv, stride_pool)
que estão sendo liberados e alocados corretamente.
Os kernels de cada camada são inicializados a partir do método he ou xavier, e os bias com valores aleatórios próximos de zero.
Os kernels e bias são atualizados com o método de otimização Adam ou SGD. É possível escolher a função de ativação, função de erro, método de pooling.
O modelo treina, avalia e reconstrói a matriz de entrada. O erro é calculado e exibido na saída.
Possui funções para configurar os parâmetros da rede, treinar, avaliar, reconstruir a matriz de entrada, exibir o espaço latente e o erro.
Testes com dados de entrada sintéticos mostraram que o modelo está aprendendo e reconstruindo a matriz de entrada.
?OBS: O OBJETO ESTÁ FECHANDO INESPERADAMENTE QUANDO É ENCERRADO E EM SEGUIDA INICIALIZADO NOVAMENTE
TODO: VERIFICAR A CAUSA DO FECHAMENTO INESPERADO DO OBJETO NO CASO ACIMA.
Fora o caso mencionado acima código está funcionando sem erros de performance aparentes. (18/02/2025)
 */

// Define uma nova classe
static t_class *cnn2d_class;

typedef struct _cnn2d {
    t_object  x_obj;

    //* variáveis para armazenar os parâmetros da rede
    t_int num_layers; //número de camadas convolucionais (encoder)
    t_int num_Tlayers; //número de camadas de convolução transposta (decoder)
    t_int simetria; //simetria entre as camadas do encoder e decoder (simetry = 1, as camadas são simétricas)
    t_int maxepochs; //número máximo de épocas
    t_int current_epoch; //época atual
    t_float learn_rate; //taxa de aprendizado
    t_float datasize; //nº de exemplos de treinamento
    t_float trainingmode_c; //modo de treinamento
    t_float evalmode; //modo de avaliação do modelo treinado
    t_float num_erro; //quantidade de vezes que a rede errou no modo de avaliação
    t_int current_data; //exemplo atual
    t_int random_state; //estado aleatório
    t_float erro_total; //erro total
    t_float beta1; //parâmetro beta1 para o otimizador Adam
    t_float beta2; //parâmetro beta2 para o otimizador Adam
    t_float epsilon; //parâmetro epsilon para o otimizador Adam (evitar divisão por zero)

    
    //* vetores para armazenar os parâmetros do ENCODER
    t_float *padding; //vetor com valores de padding para matriz de entrada de cada camada (tamanho do vetor = nº de camadas)
    t_float *pooling_matriz_size; //vetor com pares de linhas e colunas das matrizes de pooling (tamanho do vetor = 2x nº de camadas)
    t_float *input_size; //lista com tamanho dos dados de entrada (linhas e colunas) sem considerar o padding (tamanho da lista  = 2)
    t_float *input_padded; //dimensões das matrizes de entrada de cada camada (considerando o padding) (tamanho do vetor = 2x nº de camadas)
    t_float *kernels_size; //lista com pares de dimensões de cada kernel (tamanho da lista = 2x nº de camadas)
    t_float *pooling_size; //lista de dimensões da janela de pooling (não são dimensões da matriz de pooling) (tamanho do vetor = 2x nº de camadas)
    t_float *convolv_matriz_size;//lista com as dimensões das matrizes de convolução para cada camada (é calculado a partir dos parâmetros. Tamanho da lista = 2x nº de camadas)
    t_float *stride_conv;//lista com os valores de stride para linha e coluna da matriz de convolução de cada camada (tamanho do vetor = 2x nº de camadas)
    t_float *stride_pool;//lista com os valores de stride para linha e coluna do pooling de cada camada (tamanho do vetor = 2x nº de camadas)

    //* vetores para armazenar tamanhos das matrizes do DECODER
    t_float *Tpadding; //vetor com valores de padding para matriz de entrada de cada camada (tamanho do vetor = nº de camadas do decoder)
    t_float *Tconv_size; //vetor com pares de linhas e colunas das matrizes de convolução transposta (tamanho do vetor = 2x nº de camadas do decoder)
    t_float *Tinput_size; //lista com tamanho dos dados de entrada (linhas e colunas) sem considerar o padding (tamanho da lista  = 2)
    t_float *Tinput_padded; //dimensões das matrizes de entrada de cada camada (considerando o padding) (tamanho do vetor = 2x nº de camadas do decoder)
    t_float *Tstride_conv; //vetor com os valores de stride para linha e coluna da matriz de convolução transposta de cada camada (tamanho do vetor = 2x nº de camadas do decoder)
    t_float *Tkernels_size; //lista com pares de dimensão de cada kernel da convolução transposta (tamanho da lista = 2x nº de camadas do decoder)


    //* vetores do bias e dos momentos do bias
    t_float *bias_kernel; //vetor de bias para cada camada (tamanho do vetor = nº de kernels de cada camada encoder + decoder)
    t_float *m_bias; //vetor do primeiro momento do bias para cada camada (tamanho do vetor = nº de kernels de cada camada encoder + decoder)
    t_float *v_bias; //vetor do segundo momento do bias para cada camada (tamanho do vetor = nº de kernels de cada camada encoder + decoder)

    //* matrizes para a etapa de convolução ENCODER
    t_float ***input; //array de matriz de entrada (número de matriz de entrada, número de linhas, número de colunas para cada) primeira camada é a matriz de entrada original, camadas seguintes são as matrizes de pooling
    t_float ***kernels; //array de matriz de kernels (número de kernels, número de linhas, número de colunas para cada)
    t_float ***kernel_rotated; //array de matriz de kernels rotacionados (número de kernels, número de linhas, número de colunas para cada)
    t_float ***convolution; //array de matrize de convolução (número de matriz de convolução, número de linhas e número de colunas para cada)
    t_float ***pooling; //array de matriz de pooling (número de pooling, número de linhas e número de colunas para cada matriz)


    //* matrizes para a etapa de convolução transposta DECODER 
    t_float ***Tinput; //array de matriz de entrada de cada camada do decoder (número de matriz de entrada, número de linhas e número de colunas para cada) considerando padding
    t_float ***Tkernel; //array de matriz de kernels de cada camada do decoder (número de kernels transpostos, número de linhas e número de colunas para cada)
    t_float ***Tkernel_rotated; //array de matriz de kernels rotacionados de cada camada do decoder (número de kernels transpostos, número de linhas e número de colunas para cada)
    t_float ***Tconv; //array de matriz de convolução transposta de cada camada do decoder (número de matriz de convolução transposta, número de linhas e número de colunas para cada) 


    //* matrizes para backporpagation do encoder
    t_float ***delta; //array de matriz de delta de cada camada de saída (número de matriz de delta, número de linhas e número de colunas para cada) mesmo tamanho das matrizes de pooling do encoder
    t_float ***kernels_grad; //array de matriz de gradiente do kernel de cada camada do encoder (número de matriz de gradiente, número de linhas e número de colunas para cada) mesmo tamanho dos kernels do encoder
    t_float ***conv_grad; //array de matriz de gradiente da convolução (número de matriz de gradiente, número de linhas e número de colunas para cada) mesmo tamanho das matrizes de convolução do encoder

    //* matrizes para atualização dos kernels do encoder
    t_float ***m_kernel; //array de matriz do primeiro momento do kernel de cada camada (número de matriz de momento, número de linhas e número de colunas para cada) mesmo tamanho dos kernels do encoder
    t_float ***v_kernel; //array de matriz do segundo momento do kernel de cada camada (número de matriz de momento, número de linhas e número de colunas para cada) mesmo tamanho dos kernels do encoder

    //* matrizes para backporpagation do decoder
    t_float ***Tdelta; //array de matriz de delta de cada camada (número de matriz de delta, número de linhas e número de colunas para cada) mesmo tamanho das matrizes de de convolução transposta do decoder
    t_float ***Tkernel_grad; //array de matriz de gradiente do kernel de cada camada (número de matriz de gradiente, número de linhas e número de colunas para cada) mesmo tamanho dos kernels do decoder

    //* matrizes para atualização dos kernels do decoder
    t_float ***m_Tkernel; //array de matriz do primeiro momento do kernel de cada camada (número de matriz de momento, número de linhas e número de colunas para cada) mesmo tamanho dos kernels do decoder
    t_float ***v_Tkernel; //array de matriz do segundo momento do kernel de cada camada (número de matriz de momento, número de linhas e número de colunas para cada) mesmo tamanho dos kernels do decoder
    
    //* arrays de símbolos 
    t_symbol **activation_function_c; //array com nomes de funções de ativação para cada camada
    t_symbol **pooling_function; //array com nomes dos métodos de pooling para cada camada
    t_symbol *error_function; //array com nomes de funções de erro para cada camada
    t_symbol *optimizer; //array com nomes de otimização 
    
    //* buffers de saída
    t_atom *latent_out; //buffer de saída para enviar o espaço latente para outlet 1
    t_atom *matrix_out; //buffer de saída para enviar a matriz reconstruída para outlet 2
    t_atom *error_out; //buffer de saída para enviar o erro e a época para outlet 3
    t_atom *epoch_out; //buffer de saída para enviar o erro e a época para outlet 3

    //* outlets
    t_outlet  *x_out1; // Outlet 1 (espaço latente)
    t_outlet  *x_out2; // Outlet 2 (matriz reconstruída)
    t_outlet  *x_out3; // Outlet 3 (informações do aprendizado erro, época, etc)
} t_cnn2d;


//* --------------------------------  Função para gerar um número aleatório em um intervalo com estado aleatório ------------------------------------------
static double random_double_state(t_float lower, t_float upper, unsigned int *state) {
    return lower + ((double) rand_r(state) / RAND_MAX) * (upper - lower);
}


//* ----------------------------------- ativa e desativa o modo de treinamento ----------------------------------------------
static void training_mode (t_cnn2d *x, t_floatarg tra){
    // Verifica se o valor recebido é válido (0 ou 1)
    if (tra != 0 && tra != 1) {
        error("Invalid value for training mode. Use 0 for OFF and 1 for ON.");
        return;
    }

    //atualiza o estado de treinamento
    x->trainingmode_c = tra;
    int training_mode = x->trainingmode_c;

    //mensagem para indicar o estado do modo de treinamento
    switch(training_mode) {
        case 0: //desativa o modo de treinamento
            post ("training mode: OFF");
            break;
        case 1: //ativa o modo de treinamento
            x->evalmode = 0; //desativa o modo de avaliação
            post ("training mode: ON");
            break;
    }
}

//* ------------------------- taxa de aprendizado ------------------------------//
static void learning_rate(t_cnn2d *x, t_float le) {
    // Verifica se a taxa de aprendizado está no intervalo válido [0, 1]
    if (le >= 0 && le <= 1) {
        x->learn_rate = le; // Atualiza a taxa de aprendizado
        post("Learning rate: %0.4f", x->learn_rate); // Exibe a nova taxa de aprendizado
    } else {
        error("Learning rate must be between 0 and 1."); // Exibe uma mensagem de erro
    }
}


//* ------------------------- taxa de aprendizado ------------------------------//
static void random_state(t_cnn2d *x, t_float rst) {
    // Verifica se a taxa de aprendizado está no intervalo válido [0, 1]
    if (rst != A_FLOAT) {
        x->random_state = rst; // Atualiza a taxa de aprendizado
        post("Random state: %d", x->random_state); // Exibe a nova taxa de aprendizado
    } else {
        error("Random state must be integer."); // Exibe uma mensagem de erro
    }
}

//* ------------------------- ativa e desativa o modo de avaliação --------------------//
static void evalmode (t_cnn2d *x, t_floatarg eval){
    // Verifica se o valor recebido é válido (0 ou 1)
    if (eval != 0 && eval != 1) {
        error("Invalid value for evaluation mode. Use 0 for OFF and 1 for ON.");
        return;
    }

    //atualiza o estado de treinamento
    x->evalmode = eval;
    int eval_mode = x->evalmode;

    //mensagem para indicar o estado do modo de treinamento
    switch(eval_mode) {
        case 0:
            post ("evaluation mode: OFF");
            break;
        case 1: //ativa o modo de avaliação
            x->num_erro = 0; //reinicia o contador de erros
            x->current_data = 0; //reinicia o contador de exemplos
            x->trainingmode_c = 0; //desativa o modo de treinamento
            x->erro_total = 0; //reinicia o erro total
            post ("evaluation mode: ON");
            break;
    }
}

//* ----------------------------- número de épocas -----------------------------//
static void epoch_amount (t_cnn2d *x, t_floatarg ep){
    if(ep >=1 ){
        x->maxepochs = (int)ep;
        post("epochs: %d", x->maxepochs);
    } else {
        error("Amount of epochs must be greater than 0.");
    }
}


//----------------------------- número de exemplos de treinamento -----------------------------//
static void training_examples (t_cnn2d *x, t_floatarg data){
    if(data >=1 ){
        x->datasize = (int)data;
        post("Amount of training examples: %d", (int)x->datasize);
    } else {
        error("Amount of training example must be greater than 0.");
    }
}

//------------------------- ativa a simetria entre encoder e decoder --------------------//
static void symmetry_mode (t_cnn2d *x, t_floatarg simetria){
    // Verifica se o valor recebido é válido (0 ou 1)
    if (simetria != 0 && simetria != 1) {
        error("Invalid value for symmetry mode. Use 0 for OFF and 1 for ON.");
        return;
    }
    //atualiza o estado da simetria
    x->simetria = simetria;
    int simetria_mode = x->simetria;

    //mensagem para indicar o estado do modo de simetria (padrão: ON)
    if (simetria_mode == 0) {
        post("Symmetry mode: OFF. Please set the parameters for the decoder.");
    } else {
        post ("Symmetry mode: ON");
           
    }
}

//------------------------ funções de ativação --------------------------
// Definição das funções de ativação
//SIGMOIDE PARA DADOS ENTRE 0 E 1
static float sigmoid(t_float x) { 
    return 1 / (1 + exp(-x));
}

static float sigmoid_derivative(t_float x, t_float target) {
    return x * (1 - x);
}

//RELU PARA DADOS CONTÍNUOS 
static float relu(float x) {
    return x > 0 ? x : 0;
}
static float relu_derivative(t_float x, t_float target) {
    return x > 0 ? 1 : 0;
}

static float prelu(t_float x) {
    const float alpha = 0.1; // Valor fixo de alpha TODO: tornar o valor de alpha ajustável
    return x > 0 ? x : alpha * x;
}
static float prelu_derivative(t_float x, t_float target) {
    const float alpha = 0.1; // Valor fixo de alpha
    return x > 0 ? 1 : alpha;
}

//leaky ReLU PARA DADOS CONTÍNUOS
static float leaky_relu(t_float x) {
    const float alpha = 0.01; // Valor padrão de alpha para leaky ReLU
    return x > 0 ? x : alpha * x;
}

//derivada da leaky ReLU
static float leaky_relu_derivative(t_float x, t_float target) {
    const float alpha = 0.01; // Valor padrão de alpha para leaky ReLU
    return x > 0 ? 1 : alpha;
}

//TANH PARA DADOS ENTRE -1 E 1
static float tanh_activation(t_float x) {
    return tanh(x);
}

static float tanh_derivative(t_float x, t_float target) {
    return 1 - tanh(x) * tanh(x);
}

//função de ativação linear (sem ativação)
//PARA DADOS COM INTERVALOS DIFERENTES
static float linear(t_float x) {
    return x;
}
//derivada da função linear
static float linear_derivative(t_float x, t_float target) {
    return 1;
}

// Função softplus
//talvez não seja usada em cnn autoencoder
static float softplus(t_float x) {
    if (x > 20) { 
        return x; // Para valores grandes, softplus(x) ≈ x
    } else if (x < -20) {
        return exp(x); // Para valores pequenos, evita underflow
    }
    return log(1 + exp(x));
}

//*--------------------------- Derivada da função softplus --------------------------------
static float softplus_derivative(t_float x, t_float target) {
    return 1 / (1 + exp(-x)); // Derivada da função softplus é a função sigmoid
}

//* ------------------------------------- MSE ---------------------------------------------
static float mse(t_cnn2d *x, t_float p, t_float q) {
    float diff = p - q;
    return diff * diff; // (p - q)^2
}

//* --------------------------------------  DERIVADA MSE -------------------------------------
static float mse_derivative(t_cnn2d *x, t_float p, t_float q) {
    return (p - q); // 2(p - q)
}

//* --------------------------------------  MAE  ----------------------------------------------
static float mae(t_cnn2d *x, t_float p, t_float q) {
    return fabs(p - q); // |p - q|
}

//* -------------------------------------- DERIVADA MAE ----------------------------------------
static float mae_derivative(t_cnn2d *x, t_float p, t_float q) {
    float diff = p - q;
    if (diff < 0) {
        return -1;
    } else if (diff > 0) {
        return 1;
    } else {
        return 0;
    }
}

//* -------------------------------------- BCE -----------------------------------------------------
static float bce(t_cnn2d *x, t_float p, t_float q) {
    float epsilon = 1e-10;  // Para evitar log(0)
    return -(q * log(p + epsilon) + (1 - q) * log(1 - p + epsilon));
}

//* ------------------------------------ DERIVADA BCE -----------------------------------------------
static float bce_derivative(t_cnn2d *x, t_float p, t_float q) {
    float epsilon = 1e-10;
    return (p - q) / ((p + epsilon) * (1 - p + epsilon));
}


//* -------------------------------------- ATRIBUI A FUNÇÃO DE PERDA ------------------------------------------------
typedef float (*perda_func)(t_cnn2d *x, t_float, t_float);

static perda_func get_perda_function(t_symbol *func_name) {
    if (func_name == gensym("mse")) {
        return mse;
    } else if (func_name == gensym("mae")) {
        return mae;
    } else if (func_name == gensym("bce")) {
        return bce;
    } else {
        return mse; // Padrão
    }
}

//* --------------------------------------  ATRIBUI A DERIVADA DA FUNÇÃO DE PERDA --------------------------------------------
typedef float (*perda_derivada)(t_cnn2d *x, t_float, t_float);

static perda_derivada get_perda_deriv(t_symbol *func_name) {
    if (func_name == gensym("mse")) {
        return mse_derivative;
    } else if (func_name == gensym("mae")) {
        return mae_derivative;
    }else if (func_name == gensym("bce")) {
            return bce_derivative;
    } else {
        return mse_derivative; // Padrão
    }
}

//* ------------------------------------- RECEBE O NOME DA FUNÇÃO DE PERDA --------------------------------------------
static void error_function(t_cnn2d *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        error("Please provide an error function name.");
        return;
    }
    t_symbol *erro_func = atom_getsymbol(&argv[0]);
    x->error_function = erro_func;
    post("Loss function set to: %s", erro_func->s_name);
}

//*------------------------------------- função para selecionar a função de ativação e sua derivada com base no nome -----------------------------------
// Definição dos tipos de função de ativação e suas derivadas
typedef float (*activation_func)(t_float);
typedef float (*activation_derivative_func)(t_float, t_float);

// Funções para obter a função de ativação com base no nome
static activation_func get_activation_function(t_symbol *func_name) {
    if (func_name == gensym("sigmoid")) { 
        return sigmoid;
    } else if (func_name == gensym("relu")) {
        return relu;
    } else if (func_name == gensym("tanh")) {
        return tanh_activation;
    } else if(func_name == gensym("prelu")) {
        return prelu;
    } else if (func_name == gensym("softplus")){
        return softplus;
    } else if (func_name == gensym("linear")){ //para a camada de saída do encoder e decoder
        return linear;
    }else if (func_name == gensym("leakyrelu")){ //para a camada de saída do encoder e decoder
        return leaky_relu;
    } else {
        return relu; // função Padrão
    }
}

//*----------------------------------- obter a derivada da função de ativação -------------------------------------
static activation_derivative_func get_activation_derivative_function(t_symbol *func_name) {
    if (func_name == gensym("sigmoid")) {
        return sigmoid_derivative;
    } else if (func_name == gensym("relu")) {
        return relu_derivative;
    } else if (func_name == gensym("prelu")) {
        return prelu_derivative;
    } else if (func_name == gensym("tanh")) {
        return tanh_derivative;
    } else if (func_name == gensym("softplus")) {
        return softplus_derivative;
    } else if (func_name == gensym("linear")){ // para a camada de saída do encoder e decoder
        return linear_derivative;
    } else if (func_name == gensym("leakyrelu")){ // para a camada de saída do encoder e decoder
        return leaky_relu_derivative;
    } else {
        return relu_derivative; // Padrão
    }
}

//*--------------------- Função para configurar as funções de ativação de cada camada (encoder+decoder) ---------------------
static void activation_functions(t_cnn2d *x, t_symbol *s, int argc, t_atom *argv) {
    int total_layers = x->num_layers + x->num_Tlayers; //número total de camadas (encoder + decoder)
    // Verifica se a quantidade de argumentos recebidos é igual ao número total de camadas
    if (argc != total_layers) {
        error("Please provide an activation function name for each layer.");
        return;
    }
    // Atribui a função de ativação à cada camada
    for (int i = 0; i < argc; i++) {
        // Verifica se o argumento atual é um símbolo
        if (argv[i].a_type != A_SYMBOL) {
            error("Activation function is not a symbol.");
            return;
        }
        // Obtém o nome da função de ativação
        t_symbol *func_name = atom_getsymbol(&argv[i]);
        
        // armazenar o nome da função de ativação no vetor de activation_function (vetor de símbolos)
        x->activation_function_c[i] = func_name;
        post("Layer %d: activation function set to: %s", i+1, func_name->s_name);
    }
}

//*--------------------------------------------------- max pooling ---------------------------------------------------------------
static void max_pooling(t_float **matrix, int conv_rows, int conv_cols, t_float **output, int pool_rows, int pool_cols, int pool_win_rows, int pool_win_cols, int stride_rows, int stride_cols) {
    //argumentos: matriz de convolução, número de linhas e colunas, matriz de pooling, número de linhas e colunas, número de linhas e colunas da janela de pooling, stride para linha e coluna
    //Percorre cada janela de pooling
    for (int i = 0; i < pool_rows; i++) {
        int conv_i = i * stride_rows; // Posição inicial na matriz de entrada (linha)
        for (int j = 0; j < pool_cols; j++) {
            int conv_j = j * stride_cols; // Posição inicial na matriz de entrada (coluna)
            float max = -INFINITY; // Inicializa com o menor valor possível

            // Percorre a janela de pooling
            for (int k = 0; k < pool_win_rows; k++) {
                for (int l = 0; l < pool_win_cols; l++) {
                    // Verifica se a posição está dentro dos limites da matriz
                    int row = conv_i + k; // Posição atual na matriz de convolução (linha)
                    int col = conv_j + l; // Posição atual na matriz de convolução (coluna)
                    if (row < conv_rows && col < conv_cols) { // Verifica se a posição está dentro dos limites da matriz
                        if (matrix[row][col] > max) { // Verifica se o valor atual é maior que o máximo
                            max = matrix[row][col]; // Atualiza o máximo
                        }
                    }
                }
            }
            output[i][j] = max; // Armazena o máximo na matriz de saída
        }
    }
    // Retorna a matriz resultante
}

//*--------------------------------------------------- average pooling ----------------------------------------------------------------
static void avg_pooling (t_float **matriz, int conv_rows, int conv_cols, t_float **output, int pool_rows, int pool_cols, int pool_win_rows, int pool_win_cols, int stride_rows, int stride_cols){
    //argumentos: matriz de convolução, número de linhas e colunas, matriz de pooling, número de linhas e colunas, número de linhas e colunas da janela de pooling, stride para linha e coluna
    //Percorre cada janela de pooling
    for (int i = 0; i < pool_rows; i++) {
        int conv_i = i * stride_rows; // Posição inicial na matriz de entrada (linha)
        for (int j = 0; j < pool_cols; j++) {
            int conv_j = j * stride_cols; // Posição inicial na matriz de entrada (coluna)
            float sum = 0.0; // Inicializa com zero

            // Percorre a janela de pooling
            for (int k = 0; k < pool_win_rows; k++) {
                for (int l = 0; l < pool_win_cols; l++) {
                    // Verifica se a posição está dentro dos limites da matriz
                    int row = conv_i + k; // Posição atual na matriz de convolução (linha)
                    int col = conv_j + l; // Posição atual na matriz de convolução (coluna)
                    if (row < conv_rows && col < conv_cols) { // Verifica se a posição está dentro dos limites da matriz
                        sum += matriz[row][col]; // Soma o valor atual
                    }
                }
            }
            output[i][j] = sum / (pool_win_rows * pool_win_cols); // Calcula a média e armazena na matriz de saída
        }
    }
}


//*------------------------------------- seleciona o método de pooling com base no nome -----------------------------------
// Definição do tipo de função de pooling
typedef void (*pooling_func)(t_float **matriz, int conv_rows, int conv_cols, t_float **output, int pool_rows, int pool_cols, int pool_win_rows, int pool_win_cols, int stride_rows, int stride_cols);

// Função para obter o método de pooling com base no nome
static pooling_func get_pooling_method(t_symbol *func_name) {
    if (func_name == gensym("max")) {
        return max_pooling;
    } else if (func_name == gensym("avg")) {
        return avg_pooling;
    } else {
        return max_pooling; // Se func_name recebido não for válido, retorna max pooling (padrão)
    }
}

//*--------------------------------- Configura os métodos de pooling de cada camada do encoder -------------------------------------
static void pooling_methods(t_cnn2d *x, t_symbol *s, int argc, t_atom *argv) {
    // recebe o nome do método de pooling para cada camada do encoder
    // Verifica se a quantidade de argumentos é igual ao número de camadas do encoder
    if (argc != x->num_layers || argc == 0) {
        error("Please provide activation function name for each layer.");
        return;
    }
    // Atribui o método de pooling a cada camada
    for (int i = 0; i < argc; i++) { // Percorre os argumentos recebidos
        // Verifica se o argumento atual é um símbolo
        if (argv[i].a_type != A_SYMBOL) {
            error("Activation function is not a symbol.");
            return;
        }
        // Obtém o nome método de pooling de cada camada
        t_symbol *func_name = atom_getsymbol(&argv[i]);
        
        // Armazena o nome do método de pooling no vetor pooling_function (vetor de símbolos)
        x->pooling_function[i] = func_name;
        post("Pooling method for layer %d set to: %s", i+1, func_name->s_name); // Imprime o nome do método de pooling
     }
}


//*----------------------------------- STOCHASTIC GRADIENT DESCENT (SGD) -----------------------------------------------
static void sgd(t_cnn2d *x) {
    //*1. ENCODER
    //1.1. ATUALIZA OS KERNELS DE CADA CAMADA DO ENCODER 
    for(int k = 0; k < x->num_layers; k++) { //percorre cada camada do encoder
        for(int i = 0; i < x->kernels_size[k*2]; i++) {
            for(int j = 0; j < x->kernels_size[k*2+1]; j++) {
                x->kernels[k][i][j] -= x->learn_rate * x->kernels_grad[k][i][j]; //atualiza os pesos (kernels) de acordo com o gradiente descendente estocástico
            }
        }

        //![DEPURAÇÃO]
        // //imprime os kernels de cada camada do encoder
        // post("Encoder Layer %d: kernel atualizado", k+1);
        // for(int i = 0; i < x->kernels_size[k*2]; i++) {
        //     for(int j = 0; j < x->kernels_size[k*2+1]; j++) {
        //         post("%0.4f", x->kernels[k][i][j]);
        //     }
        //     post("");
        // }
        //![DEPURAÇÃO]
        
        //1.2. ATUALIZA OS BIAS DE CADA CAMADA DO ENCODER
        float bias_grad = 0.0; //inicializa o gradiente do bias com zero
        for(int i = 0; i < x->convolv_matriz_size[k*2]; i++) {
            for(int j = 0; j < x->convolv_matriz_size[k*2+1]; j++) {
                bias_grad += x->conv_grad[k][i][j]; //soma todos os valores de gradiente da convolução de cada camada do encoder
            }
        }
        x->bias_kernel[k] -= x->learn_rate * bias_grad; //atualiza o bias de acordo com o gradiente descendente estocástico

        //![DEPURAÇÃO]
        // //imprime o bias de cada camada do encoder
        // post("Encoder Layer %d: bias atualizado", k+1);
        // post("%0.4f", x->bias_kernel[k]);
        //![DEPURAÇÃO]
    }

    //* 2. DECODER
    //2.1. ATUALIZA OS KERNELS DE CADA CAMADA DO DECODER
    for(int k = 0; k < x->num_Tlayers; k++) { //percorre cada camada do decoder
        for(int i = 0; i < x->Tkernels_size[k*2]; i++) { //percorre as linhas do kernel
            for(int j = 0; j < x->Tkernels_size[k*2+1]; j++) {//percorre as colunas do kernel
                x->Tkernel[k][i][j] -= x->learn_rate * x->Tkernel_grad[k][i][j]; //atualiza os pesos (kernels) do decoder
            }
        }

        //![DEPURAÇÃO]
        // //imprime os kernels de cada camada do decoder
        // post("Decoder Layer %d: kernel atualizado", k+1);
        // for(int i = 0; i < x->Tkernels_size[k*2]; i++) {
        //     for(int j = 0; j < x->Tkernels_size[k*2+1]; j++) {
        //         post("%0.4f", x->Tkernel[k][i][j]);
        //     }
        //     post("");
        // }
        //![DEPURAÇÃO]
        
        //2.2. ATUALIZA OS BIAS DE CADA CAMADA DO DECODER
        float T_bias_grad = 0.0; //inicializa o gradiente do bias com zero
        for(int i = 0; i < x->Tconv_size[k*2]; i++) { //percorre as linhas da matriz de delta da convolução transposta (saída de cada camada do decoder)
            for(int j = 0; j < x->Tconv_size[k*2+1]; j++) { //percorre as colunas da matriz de delta da convolução transposta (saída de cada camada do decoder)
                T_bias_grad += x->Tdelta[k][i][j]; //soma todos os valores de delta de cada camada do decoder (gradiente do bias)
            }
        }
        x->bias_kernel[k + x->num_layers] -= x->learn_rate * T_bias_grad; //atualiza o bias (ATENÇÃO!!! o bias do decoder começa a partir da última camada do encoder)

        //![DEPURAÇÃO]
        // //imprime o bias de cada camada do decoder
        // post("Decoder Layer %d: bias atualizado", k+1);
        // post("%0.4f", x->bias_kernel[k + x->num_layers]);
        //![DEPURAÇÃO]
    }
}

//*------------------------------------------ ADAPTIVE MOMENT ESTIMATION (ADAM) ------------------------------------------
//ver https://arxiv.org/pdf/1412.6980.pdf
static void adam(t_cnn2d *x, int epoch_atual) {

    int epoch = epoch_atual + 1; //incrementa o número da época atual para começar a partir de 1

    //* 1. ATUALIZAÇÃO DOS PARÂMETROS DO ENCODER
    for(int k = 0; k < x->num_layers; k++) {
        // 1.1 Atualiza kernels do encoder
        for(int i = 0; i < x->kernels_size[k*2]; i++) {
            for(int j = 0; j < x->kernels_size[k*2+1]; j++) {
                // Atualiza momentos
                x->m_kernel[k][i][j] = x->beta1 * x->m_kernel[k][i][j] + (1 - x->beta1) * x->kernels_grad[k][i][j];  //primeiro momento do kernel
                x->v_kernel[k][i][j] = x->beta2 * x->v_kernel[k][i][j] + (1 - x->beta2) * pow(x->kernels_grad[k][i][j], 2); //segundo momento do kernel
                
                //Calcula bias-corrected estimates
                float m_hat = x->m_kernel[k][i][j] / (1 - pow(x->beta1, epoch));
                float v_hat = x->v_kernel[k][i][j] / (1 - pow(x->beta2, epoch));
                
                // Atualiza kernels
                x->kernels[k][i][j] -= x->learn_rate * m_hat / (sqrt(v_hat) + x->epsilon);
            }
        }

        //![DEPURAÇÃO
        // //imprime os kernels de cada camada do encoder
        // post("Encoder Layer %d: kernel atualizado", k+1);
        // for(int i = 0; i < x->kernels_size[k*2]; i++) {
        //     for(int j = 0; j < x->kernels_size[k*2+1]; j++) {
        //         post("%0.4f", x->kernels[k][i][j]);
        //     }
        //     post("");
        // }
        //![DEPURAÇÃO
        
        // 1.2 Atualiza bias do encoder
        float bias_grad = 0.0;
        for(int i = 0; i < x->convolv_matriz_size[k*2]; i++) {
            for(int j = 0; j < x->convolv_matriz_size[k*2+1]; j++) {
                bias_grad += x->conv_grad[k][i][j];
            }
        }
        
        // Atualiza os momentos do bias
        x->m_bias[k] = x->beta1 * x->m_bias[k] + (1 - x->beta1) * bias_grad; //primeiro momento do bias
        x->v_bias[k] = x->beta2 * x->v_bias[k] + (1 - x->beta2) * pow(bias_grad, 2); //segundo momento do bias
        
        // Calcula bias-corrected estimates
        float m_hat_bias = x->m_bias[k] / (1 - pow(x->beta1, epoch)); //bias-corrected primeiro momento
        float v_hat_bias = x->v_bias[k] / (1 - pow(x->beta2, epoch)); //bias-corrected segundo momento
        
        // Atualiza bias
        x->bias_kernel[k] -= x->learn_rate * m_hat_bias / (sqrt(v_hat_bias) + x->epsilon);
    }
    
    //* 2. ATUALIZAÇÃO DOS PARÂMETROS DO DECODER
    for(int k = 0; k < x->num_Tlayers; k++) {
        // 2.1 Atualiza kernels do decoder
        for(int i = 0; i < x->Tkernels_size[k*2]; i++) {
            for(int j = 0; j < x->Tkernels_size[k*2+1]; j++) {
                // Atualiza momentos
                x->m_Tkernel[k][i][j] = x->beta1 * x->m_Tkernel[k][i][j] + (1 - x->beta1) * x->Tkernel_grad[k][i][j]; //primeiro momento do kernel
                x->v_Tkernel[k][i][j] = x->beta2 * x->v_Tkernel[k][i][j] + (1 - x->beta2) * pow(x->Tkernel_grad[k][i][j], 2); //segundo momento do kernel
                
                // Calcula bias-corrected estimates
                float m_hat = x->m_Tkernel[k][i][j] / (1 - pow(x->beta1, epoch));
                float v_hat = x->v_Tkernel[k][i][j] / (1 - pow(x->beta2, epoch));
                
                // Atualiza pesos
                x->Tkernel[k][i][j] -= x->learn_rate * m_hat / (sqrt(v_hat) + x->epsilon);
            }
        }

        //![DEPURAÇÃO]
        // //imprime os kernels de cada camada do decoder
        // post("Decoder Layer %d: kernel atualizado", k+1);
        // for(int i = 0; i < x->Tkernels_size[k*2]; i++) {
        //     for(int j = 0; j < x->Tkernels_size[k*2+1]; j++) {
        //         post("%0.4f", x->Tkernel[k][i][j]);
        //     }
        //     post("");
        // }
        //![DEPURAÇÃO]
        
        // 2.2 Atualiza bias do decoder
        float T_bias_grad = 0.0;
        for(int i = 0; i < x->Tconv_size[k*2]; i++) {
            for(int j = 0; j < x->Tconv_size[k*2+1]; j++) {
                T_bias_grad += x->Tdelta[k][i][j];
            }
        }
        
        // Atualiza momentos do bias
        x->m_bias[k + x->num_Tlayers] = x->beta1 * x->m_bias[k + x->num_Tlayers] + (1 - x->beta1) * T_bias_grad;
        x->v_bias[k + x->num_Tlayers] = x->beta2 * x->v_bias[k + x->num_Tlayers] + (1 - x->beta2) * pow(T_bias_grad, 2);
        
        // Calcula bias-corrected estimates
        float m_hat_Tbias = x->m_bias[k + x->num_Tlayers] / (1 - pow(x->beta1, epoch));
        float v_hat_Tbias = x->v_bias[k + x->num_Tlayers] / (1 - pow(x->beta2, epoch));
        
        // Atualiza bias do decoder
        x->bias_kernel[k + x->num_Tlayers] -= x->learn_rate * m_hat_Tbias / (sqrt(v_hat_Tbias) + x->epsilon);
    }

    //![DEPURAÇÃO]
    // //imprime o bias de cada camada do encoder e decoder
    // for(int k = 0; k < x->num_layers + x->num_Tlayers; k++) {
    //     post("bias atualizado", k+1);
    //     post("%0.4f", x->bias_kernel[k]);
    // }
    //![DEPURAÇÃO]
}


//*---------------------------------------------- SELECIONA OTIMIZADOR ---------------------------------------------------
static void optimizer(t_cnn2d *x, t_symbol *s, int argc, t_atom *argv) {
    //1. Verifica se o argumento recebido é um símbolo (nome do otimizador) e se a quantidade de argumentos é igual a 1
    if (argc != 1 || argv[0].a_type != A_SYMBOL) {
        error("Please provide an optimizer name.");
        return;
    }
    //2. Obtém o nome do otimizador
    t_symbol *optimizer_name = atom_getsymbol(&argv[0]);
    //3. Armazena o nome do otimizador na variável optimizer
    x->optimizer = optimizer_name;

    x->beta1 = 0.9; //beta1
    x->beta2 = 0.999; //beta2
    x->epsilon = 1e-8; //epsilon
    post("Optimizer set to: %s", optimizer_name->s_name);
}


//*------------------------------------------ LIBERA MEMÓRIA DE UM VETOR -------------------------------------------
static void liberar_vetor(t_cnn2d *x, t_float **vetor) {
    if (*vetor != NULL) { // Verifica se o vetor não é nulo (se não for nulo significa que já foi alocado e então pode ser liberado)
        free(*vetor);     // Libera a memória alocada para o vetor
        *vetor = NULL;    // Garante que o ponteiro seja nulo após liberar
    }
}

//* ------------------------------------ ALOCA MEMÓRIA PARA UM VETOR -------------------------------------------------
static int alocar_vetor(t_cnn2d *x, t_float **vetor, int tamanho) { //argumentos: ponteiro para a estrutura, ponteiro para o vetor, tamanho do vetor (tamanho do vetor está relacionado ao nº de camadas)
    //verifica se o vetor já foi alocado ou se o tamanho é inválido
    if (*vetor != NULL || tamanho <= 0) { //para alocar memória, o vetor deve ser NULL (o que significa que sua memória já foi liberada e está pronto para ser alocado) e tamanho deve ser maior que 0
        error("Vector is already allocated or invalid size or size is less than or equal to 0");
        return 0; // Retorna 0 para indicar falha
    }
    // Aloca memória para o vetor
    *vetor = (t_float *)malloc(tamanho * sizeof(t_float)); // Aloca memória para o vetor com o tamanho especificado
    
    // Verifica se a alocação foi bem-sucedida
    if (*vetor == NULL) {
        error("Error allocating memory for vector");
        liberar_vetor(x, vetor); // Libera o vetor alocado
        return 0; // Retorna 0 para indicar falha
    }
    // post("Vetor alocado com tamanho: %d", tamanho);
    return 1; // Retorna 1 para indicar sucesso
}

//*----------------------------------- LIBERA MEMÓRIA DO VETOR DE BIAS -------------------------------------------
static void free_bias(t_cnn2d *x, t_float **bias) {
    post("ENTRANDO EM FREE BIAS"); // Debug
    if (*bias != NULL) { // Verifica se o vetor não é nulo (se não for nulo significa que já foi alocado e então pode ser liberado)
        free(*bias);     // Libera a memória alocada para o vetor
        *bias = NULL; 
        post("BIAS FOI LIBERADO"); // Debug   // Garante que o ponteiro seja nulo após liberar
    }
}

//*------------------------------------ LIBERA MEMÓRIA DO VETOR DE MOMENTO DO BIAS ------------------------------------
static void free_momtento_bias(t_cnn2d *x, t_float **vetor) {
    if (*vetor != NULL) { // Verifica se o vetor não é nulo (se não for nulo significa que já foi alocado e então pode ser liberado)
        free(*vetor);     // Libera a memória alocada para o vetor
        *vetor = NULL;    // Garante que o ponteiro seja nulo após liberar
    }
}

//*----------------------------- ALOCA MEMÓRIA PARA O VETOR DE MOMENTO DO BIAS ---------------------------------------
static int alocar_momento_bias(t_cnn2d *x, t_float **vetor, int size) { //argumentos: ponteiro para a estrutura, ponteiro para o vetor, tamanho do vetor (tamanho do vetor está relacionado ao nº de camadas)
    //verifica se o vetor já foi alocado ou se o tamanho é inválido
    if (*vetor != NULL || size <= 0) { //para alocar memória, o vetor deve ser NULL (o que significa que sua memória já foi liberada e está pronto para ser alocado) e tamanho deve ser maior que 0
        error("Vector is already allocated or invalid size or size is less than or equal to 0");
        return 0; // Retorna 0 para indicar falha
    }
    // Aloca memória para o vetor
    *vetor = (float *)calloc(size, sizeof(float)); // Aloca memória para o vetor e inicializa com zero (calloc)
    
    // Verifica se a alocação foi bem-sucedida
    if (*vetor == NULL) {
        error("Error allocating memory for vector");
        free_momtento_bias(x, vetor); // Libera o vetor alocado
        return 0; // Retorna 0 para indicar falha
    }
    // post("Vetor alocado com tamanho: %d", tamanho);
    return 1; // Retorna 1 para indicar sucesso
}

//-----------------------------Função para alocar memória para vetor de bias ---------------------------------------
static int alocar_bias(t_cnn2d *x, t_float **bias, int size) { //argumentos: ponteiro para a estrutura, ponteiro para o vetor, tamanho do vetor (tamanho do vetor está relacionado ao nº de camadas)
    //verifica se o vetor já foi alocado ou se o tamanho é inválido
    if (*bias != NULL || size <= 0) { //para alocar memória, o vetor deve ser NULL (o que significa que sua memória já foi liberada e está pronto para ser alocado) e tamanho deve ser maior que 0
        error("Bias vector is already allocated or invalid size or size is less than or equal to 0");
        return 0; // Retorna 0 para indicar falha
    }
    // Aloca memória para o vetor
    *bias = (float *)malloc(size * sizeof(float)); // Aloca memória para o vetor com o tamanho especificado
    
    // Verifica se a alocação foi bem-sucedida
    if (*bias == NULL) {
        error("Error allocating memory for bias vector");
        free_bias(x, bias); // Libera o vetor alocado
        return 0; // Retorna 0 para indicar falha
    }
    post("Bias vector: %d", size);
    return 1; // Retorna 1 para indicar sucesso
}

//------------------------- função para preencher o vetor de bias com valores aleatórios ---------------------------------------
static void bias_fill(t_cnn2d *x, t_float *bias, int size, t_float lower, t_float upper, unsigned int *state) {
    if (bias == NULL) { // Verifica se o vetor é nulo
        error("Bias vector is not allocated. Please allocate memory for the bias vector.");
        return;
    }
    // int total_layers = x->num_layers + x->num_Tlayers; //número total de camadas (encoder + decoder)    
    for (int i = 0; i < size ; i++) { // Preenche o vetor de bias com valores aleatórios
        bias[i] = random_double_state(lower, upper, state);
        post("Bias %d: %.6f", i, x->bias_kernel[i]);
    }
}

//------------------------- função para redimensionar um vetor ---------------------------------------
static int ajustar_vetor(t_cnn2d *x, t_float **vetor, int tamanho, const char *erro_msg) {
    liberar_vetor(x, vetor); // Libera a memória atual
    if (!alocar_vetor(x, vetor, tamanho)) { // Realoca com o novo tamanho
        error("%s", erro_msg);
        return 0; // Indica falha
    }
    return 1; // Indica sucesso
}

//*------------------------- LIBERA MEMÓRIA DAS MATRIZES  ---------------------------------------
//obs: a função está aceitando double ****matriz e *size como argumento (está funcionando corretamente)
static void free_matrix(t_cnn2d *x, t_float ****matriz, int num_matrizes, t_float *size) {
// argumentos: ponteiro para a estrutura, ponteiro para o array de matrizes, número de matrizes, ponteiro para o vetor de tamanhos das matrizes
    if (size == NULL || num_matrizes <= 0) { // Verifica se o vetor de tamanhos e o número de matrizes são válidos
        error("Size vector or number of matrices is invalid(free)");
        return; // Nada a liberar
    }
    if (*matriz != NULL) { // Para liberar memória, a matriz deve ser diferente de NULL
        for (int m = 0; m < num_matrizes; m++) {
            if ((*matriz)[m] != NULL) { // Verifica se a matriz m é diferente de NULL
                int linhas = (int)size[m * 2]; // Recupera o número de linhas da matriz m
                for (int i = 0; i < linhas; i++) { // Percorre as linhas da matriz m
                    if ((*matriz)[m][i] != NULL) { // Verifica se a linha i da matriz m é diferente de NULL
                        free((*matriz)[m][i]); // Libera a memória da linha i da matriz m
                        (*matriz)[m][i] = NULL; // Define a linha i da matriz m como NULL
                    }
                }
                free((*matriz)[m]); // Libera a memória da matriz m
                (*matriz)[m] = NULL; // Define a matriz m como NULL
            }
        }
        free(*matriz); // Libera a memória do array de matrizes
        *matriz = NULL; // Define o array de matrizes como NULL
    }
}
    
//*------------------------------ ALOCA MEMÓRIA PARA AS MATRIZES ---------------------------------------
//obs: a função está aceitando double ****matriz e *size como argumento (está funcionando corretamente)
static int alocar_matrizes(t_cnn2d *x, t_float ****matriz, int num_matrizes, t_float *size) {
    if (*matriz != NULL) { // Verifica se a matriz já foi alocada
        error("Matrices have already been allocated. Set the parameters and use the 'create' message.");
        return 0; // Retorna 0 para indicar falha
    }
    if (size == NULL || num_matrizes <= 0) { // Verifica se o vetor de tamanhos e o número de matrizes são válidos
        error("Size vector and number of matrices is invalid(alocar)");
        return 0; 
    }
    for (int i = 0; i < num_matrizes * 2; i++) {// Percorre o vetor de dimensões
        if ((int)size[i] <= 0) {// Verifica se os tamanhos das matrizes são válidos
            error("Invalid size for matrix %d. Size must be positive integer", i / 2 + 1);
            return 0; 
        }
    }
    *matriz = (t_float ***)malloc(num_matrizes * sizeof(t_float **)); // Aloca memória para o array de matrizes
    if (*matriz == NULL) {// Verifica se a alocação foi bem-sucedida
        error("Error allocating memory for the array of matrices");
        return 0;
    }
    for (int m = 0; m < num_matrizes; m++) {// Percorre as matrizes
        int linhas = (int)size[m * 2];// Recupera o número de linhas da matriz m
        int colunas = (int)size[m * 2 + 1];// Recupera o número de colunas da matriz m
        (*matriz)[m] = (t_float **)malloc(linhas * sizeof(t_float *)); // Aloca memória para a matriz m com o tamanho de linhas
        if ((*matriz)[m] == NULL) { // Verifica se a alocação foi bem-sucedida
            error("Error allocating memory for rows of matrix %d", m);
            free_matrix(x, matriz, num_matrizes, size); // Libera a memória alocada se houver erro
            return 0;
        }
        for (int l = 0; l < linhas; l++) {// Percorre as linhas da matriz m
            (*matriz)[m][l] = (t_float *)malloc(colunas * sizeof(t_float)); // Aloca memória para a linha l da matriz m com o tamanho de colunas
            if ((*matriz)[m][l] == NULL) { // Verifica se a alocação foi bem-sucedida
                error("Error allocating memory for row %d of matrix %d", l, m);
                free_matrix(x, matriz, num_matrizes, size); // Libera a memória alocada se houver erro
                return 0;
            }
        }
        
    }
    return 1;
}


//*----------------------------- PREENCHE AS MATRIZES -------------------------------------
static void matriz_fill(t_cnn2d *x, t_float ***matriz, t_int num_matrizes, t_float *size, t_float lower, t_float higher, unsigned int *state, t_symbol *name) {
    //argumentos: matriz, nº de matrizes, dimensões da matriz, valor mínimo, valor máximo
    //verifica se a matriz é nula
   if (matriz == NULL) {
        error("Pointer to matrices is NULL. Cannot fill matrices.");
        return;
    }
    //verifica se o tamanho do vetor e o número de matrizes é inválido
    if (size == NULL || num_matrizes <= 0) {
        error("Size vector or number of matrices is invalid. Cannot fill matrices.");
        return;
    }
    srand(time(NULL)); // Inicia a semente para gerar números aleatórios
    // Preenche cada matriz com valores aleatórios
    for (int m = 0; m < num_matrizes; m++) { //percorre as matrizes
        int linhas = (int)size[m * 2]; //recupera o nº de linhas da matriz armazenado no vetor de dimensões
        int colunas = (int)size[m * 2 + 1]; //recupera o nº de colunas da matriz armazenado no vetor de dimensões
        //verifica se as dimensões da matriz são válidas
        if (linhas <= 0 || colunas <= 0) {
            error("Invalid dimensions for matrix %d. Skipping.", m + 1);
            continue;
        }
        post("Layer %d: matrix: %s", m + 1, name->s_name); //imprime as dimensões da matriz
        for (int i = 0; i < linhas; i++) { //percorre as linhas da matriz
            for (int j = 0; j < colunas; j++) { //percorre as colunas da matriz
                matriz[m][i][j] = random_double_state(lower, higher, state); // Preenche com valores aleatórios de acordo com o intervalo
                post("Matriz %d: linha %d: %0.2f ", m+1, i+1, matriz[m][i][j]); // Imprime o valor com um espaço para facilitar a leitura
            }
        }
        post(""); // Quebra de linha entre as matrizes
    }
} 


//*---------------------------------- INICIALIZAÇÃO HE ---------------------------------------
// ver He et al. (2015) [https://arxiv.org/abs/1502.01852] método utilizado com funções de ativação ReLU
static void random_he(t_cnn2d *x) {
    unsigned int state = (unsigned int)x->random_state; // Inicializa a semente para gerar números aleatórios
    int total_layers = x->num_layers + x->num_Tlayers; //número total de camadas (encoder + decoder)

    //CALCULA LIMITES DOS KENRNELS DO ENCODER
    float encoder_lower; // Limite inferior para os valores dos kernels
    float encoder_higher; // Limite superior para os valores dos kernels
    for (int m = 0; m < x->num_layers; m++) {
        int kernel_size = x->kernels_size[m * 2] * x->kernels_size[m * 2 + 1]; //dimensões do kernel (nº de linhas x nº de colunas)
        // Calcula os limites inferior e superior para os valores dos kernels
        encoder_lower = -sqrt(2.0 / kernel_size);
        encoder_higher = sqrt(2.0 / kernel_size);
        post("Encoder Layer %d: Kernels initialized between %0.2f and %0.2f.", m + 1, encoder_lower, encoder_higher);
    }
    // Preenche os kernels com valores aleatórios
    matriz_fill(x, x->kernels, x->num_layers, x->kernels_size, encoder_lower, encoder_higher, &state, gensym("Encoder kernel")); // inicializa os kernels do encoder

     //CALCULA LIMITES DO KENRNELS DO DECODER
    float decoder_lower; // Limite inferior para os valores dos kernels
    float decoder_higher; // Limite superior para os valores dos kernels
    for (int m = 0; m < x->num_Tlayers; m++) {
        int kernel_size = x->Tkernels_size[m * 2] * x->Tkernels_size[m * 2 + 1]; //dimensões do kernel (nº de linhas x nº de colunas)
        // Calcula os limites inferior e superior para os valores dos kernels
        decoder_lower = -sqrt(2.0 / kernel_size);
        decoder_higher = sqrt(2.0 / kernel_size);
        post("Decoder Layer %d: Kernels initialized between %0.2f and %0.2f.", m + 1, decoder_lower, decoder_higher);
    }
     // Preenche os kernels com valores aleatórios
    matriz_fill(x, x->Tkernel, x->num_Tlayers, x->Tkernels_size, decoder_lower, decoder_higher, &state, gensym("Decoder kernel")); // inicializa os kernels do decoder

    post("Bias vectors initialized between 0 and 0.3.");
    // Preenche os bias com valores aleatórios
    bias_fill(x, x->bias_kernel, total_layers, 0, 0.3, &state); //inicializa os bias
}

//*------------------------------------- INICIALIZAÇÃO XAVIER --------------------------------------------
// ver Glorot e Bengio (2010) [http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf] método utilizado com funções de ativação sigmoid e tanh
static void random_xavier(t_cnn2d *x) {
    unsigned int state = (unsigned int)x->random_state; // Inicializa a semente para gerar números aleatórios
    int total_layers = x->num_layers + x->num_Tlayers; //número total de camadas (encoder + decoder)

    //* CALCULA LIMITES DOS KENRNELS DO ENCODER
    float encoder_lower; // Limite inferior para os valores dos kernels
    float encoder_higher; // Limite superior para os valores dos kernels
    for (int m = 0; m < x->num_layers; m++) {
        int input_neurons = x->kernels_size[m * 2] * x->kernels_size[m * 2 + 1];
        int output_neurons = x->convolv_matriz_size[m * 2] * x->convolv_matriz_size[m * 2 + 1];
        
        encoder_lower = -sqrt(6.0 / (input_neurons + output_neurons));
        encoder_higher = sqrt(6.0 / (input_neurons + output_neurons));
        post("Encoder Layer %d: Kernels initialized between %0.2f and %0.2f.", m + 1, encoder_lower, encoder_higher);
    }
    matriz_fill(x, x->kernels, x->num_layers, x->kernels_size, encoder_lower, encoder_higher, &state, gensym("Encoder kernel")); // inicializa os kernels do encoder

    //* CALCULA LIMITES DOS KENRNELS DO DECODER
    float decoder_lower; // Limite inferior para os valores dos kernels
    float decoder_higher; // Limite superior para os valores dos kernels
    for (int m = 0; m < x->num_Tlayers; m++) {
        int input_neurons = x->Tkernels_size[m * 2] * x->Tkernels_size[m * 2 + 1];
        int output_neurons = x->Tconv_size[m * 2] * x->Tconv_size[m * 2 + 1];
        
        decoder_lower = -sqrt(6.0 / (input_neurons + output_neurons));
        decoder_higher = sqrt(6.0 / (input_neurons + output_neurons));
        post("Decoder Layer %d: Kernels initialized between %0.2f and %0.2f.", m + 1, decoder_lower, decoder_higher);
    }
    matriz_fill(x, x->Tkernel, x->num_Tlayers, x->Tkernels_size, decoder_lower, decoder_higher, &state, gensym("Decoder kernel")); // inicializa os kernels do decoder

    bias_fill(x, x->bias_kernel, total_layers, 0, 0.3, &state); // inicializa os bias
    post("Bias vectors initialized between 0 and 0.3.");
}
  
//*--------------------------------------- CALCULA AS DIMENSÕES DAS MATRIZES DO ENCODER ---------------------------------------------
static void matrix_encoder_size(t_cnn2d *x) {
    //esta função é chamada após a definição dos parâmetros da rede (nº de camadas, dimensões dos dados de entrada, dimensões dos kernels, padding, stride, etc)
    if (x->num_layers <= 0) { // Verifica se o número de camadas é válido
        error("Invalid number of encoder layers");
        return;
    }
    post("Encoder layers: %d", x->num_layers);

    //----------------------------------------- libera memória das matrizes do encoder ----------------------------------------------------------
    //ATENÇÃO: É IMPORTANTE LIBERAR MEMÓRIA DAS MATRIZES ANTES DE MUDAR AS DIMENSÕES PARA EVITAR VAZAMENTO DE MEMÓRIA
    //obs: as matrizes de kernel são liberadas e alocadas na função kernel_size porque só dependem do número de camadas e das dimensões dos kernels
    //argumentos: matriz, nº de matrizes, vetor de tamanho da matriz de entrada
    free_matrix(x, &x->input, x->num_layers, x->input_padded); //MATRIZ DE ENTRADA 
    free_matrix(x, &x->convolution, x->num_layers, x->convolv_matriz_size); //MATRIZ DE CONVOLUÇÃO 
    free_matrix(x, &x->pooling, x->num_layers, x->pooling_matriz_size); //MATRIZ DE POOLING
    free_matrix(x, &x->delta, x->num_layers, x->pooling_matriz_size); //MATRIZ DE DELTA DA SAÍDA DE CADA CAMADA DO ENCODER
    free_matrix(x, &x->conv_grad, x->num_layers, x->convolv_matriz_size); //MATRIZ DE GRADIENTE DA CONVOLUÇÃO 

   //------------------------------- CALCULA AS DIMENSÕES DA MATRIZ DE CONVOLUÇÃO E DE POOLING (ENCODER) -----------------------------------
    for (int i = 0; i < x->num_layers; i++) { //percorre as camadas do encoder
        //------------------------------- RECUPERA DIMENSÕES DA MATRIZ DE ENTRADA SEM PADDING -----------------------------------
        // Se for a primeira camada do encoder, a matriz de entrada é a matriz de dados, caso contrário, é a matriz de pooling da camada anterior
        //considerar a matriz de entrada sem pooling somente para o calculo das dimensões da matriz de convolução e pooling
        int input_rows = (i == 0) ? (int)x->input_size[0] : (int)x->pooling_matriz_size[(i - 1) * 2];
        int input_cols = (i == 0) ? (int)x->input_size[1] : (int)x->pooling_matriz_size[(i - 1) * 2 + 1];
        // post("Layer %d: input matrix - %d x %d", i +1, input_rows, input_cols);

        //------------------------------------- RECUPERA PADDING ---------------------------------------------
        int padding = (int)x->padding[i]; //padding da matriz de entrada
       
        //----------------------------------- RECUPERA DIMENSÕES DOS KERNELS -----------------------------------
        int kernel_rows = (int)x->kernels_size[i * 2];
        int kernel_cols = (int)x->kernels_size[i * 2 + 1];
        
        //------------------------------ RECUPERA DIMENSÕES DOS STRIDES DA CONVOLUÇÃO ------------------------------
        int stride_conv_rows = (int)x->stride_conv[i * 2]; //stride da linha (convolução)
        int stride_conv_cols = (int)x->stride_conv[i * 2 + 1]; //stride da coluna (convolução)
        
        //----------------------------- RECUPERA DIMENSÕES DA JANELA DE POOLING -----------------------------------
        int pooling_rows = (int)x->pooling_size[i * 2]; //linha da janela de pooling
        int pooling_cols = (int)x->pooling_size[i * 2 + 1];//coluna da janela de pooling
        
        //------------------------------- RECUPERA DIMSENSÕES DOS STRIDES DO POOLING ----------------------------------
        int stride_pool_rows = (int)x->stride_pool[i * 2];
        int stride_pool_cols = (int)x->stride_pool[i * 2 + 1];
        
        //------------------------------- VERIFICA SE OS VALORES DE KERNEL, STRIDE (CONV E POOL) E JANELA DE POOLING SÃO VÁLIDOS -----------------------------
        if (stride_conv_rows <= 0 || stride_conv_cols <= 0 || 
            stride_pool_rows <= 0 || stride_pool_cols <= 0 || 
            kernel_rows <= 0 || kernel_cols <= 0 || 
            pooling_rows <= 0 || pooling_cols <= 0) {
            error("Layer %d: invalid values for stride, kernel size, or pooling window", i + 1);
            return;
        }

        //*------------------------------- VERIFICA SE O STRIDE É VÁLIDO EM RELAÇÕA AO KERNEL ----------------------------------------
        //stride deve ser menor ou igual ao kernel para evitar lacunas
        if(stride_conv_rows > kernel_rows || stride_conv_cols > kernel_cols){
            error("Layer %d: convolution stride (%d x %d) must be less than or equal kernel (%d x %d)", i + 1, stride_conv_rows, stride_conv_cols, kernel_rows, kernel_cols);
            return;
        }

        //*------------------------------- VERIFICA SE AS DIMENSÕES DO KERNEL SÃO VÁLIDAS EM RELAÇÃO AO STRIDE DA CONVOLUÇÃO, À MATRIZ DE ENTRADA E AO PADDING ----------------------
        if(kernel_rows + stride_conv_rows > input_rows + 2 * padding || kernel_cols + stride_conv_cols > input_cols + 2 * padding){
            error("Layer %d: kernel (%d x %d) + convolution stride (%d x %d) must be less than or equal input matrix (%d x %d) + 2 x padding (%d)", 
                i +1, kernel_rows, kernel_cols, stride_conv_rows, stride_conv_cols, input_rows, input_cols, padding);
            return;
        }
        
        //*-------------------------------------- CALCULA O TAMANHO DA MATRIZ DE CONVOLUÇÃO DE CADA CAMADA ---------------------------------------------------
        
        int conv_rows = (int)((input_rows - kernel_rows + 2 * padding) / stride_conv_rows) + 1;
        int conv_cols = (int)((input_cols - kernel_cols + 2 * padding) / stride_conv_cols) + 1;
        //verifica se as dimensões da matriz de convolução são válidas
        if(conv_rows <= 0 || conv_cols <= 0){
            error("Layer %d: convolution matrix (%d x %d) is invalid", i + 1, conv_rows, conv_cols);
            return;
        }
        else{
            x->convolv_matriz_size[i * 2] = conv_rows; //linhas da matriz de convolução
            x->convolv_matriz_size[i * 2 + 1] = conv_cols; //colunas da matriz de convolução
        }
        
        //*------------------------------------------ VERIFICA SE O STRIDE DO POOLING É VÁLIDO ---------------------------------------------------------
        //stride do pooling deve ser menor ou igual à matriz de convolução
        if (stride_pool_rows > conv_rows || stride_pool_cols > conv_cols) {
            error("Layer %d: pooling stride (%d x %d) exceeds convolution matrix (%d x %d)", 
                  i + 1, stride_pool_rows, stride_pool_cols, conv_rows, conv_cols);
            return;
        }
    
        //*----------------------------------- VERIFICA SE AS DIMENSÕES DA JANELA DE POOLING SÃO VÁLIDAS EM RELAÇÃO AO STRIDE DO POOLING E À MATRIZ DE CONVOLUÇÃO --------------------
        if(pooling_rows + stride_pool_rows > conv_rows || pooling_cols + stride_pool_cols > conv_cols){
            error("Layer %d: pooling window (%d x %d) + pooling stride (%d x %d) must be less tha or equal convolution matrix (%d x %d)", 
                i +1, pooling_rows, pooling_cols, stride_pool_rows, stride_pool_cols, conv_rows, conv_cols);
                return;
        }

        //*--------------------------------- VERIFICA SE O STRIDE DO POOLING É MENOR QUE A JANELA DE POOLING ---------------------------------------------
        //stride do pooling deve ser menor ou igual à janela de pooling
        if(stride_pool_rows > pooling_rows || stride_pool_cols > pooling_cols){
            error("Layer %d: pooling stride (%d x %d) must be less than or equal pooling window (%d x %d)", i +1, stride_pool_rows, stride_pool_cols, pooling_rows, pooling_cols);
            return;
        }

        //*-------------------------------------- VERIFICA SE A MATREZ DE POOLING SERÁ VÁLIDA (MAIOR QUE 0) -------------------------------------------
        if ((conv_rows - stride_pool_rows) < 0 || (conv_cols - stride_pool_cols) < 0) {
            error("Layer %d: pooling stride (%d x %d)is too large for convolution matrix (%d x %d)", i + 1, stride_pool_rows, stride_pool_cols, conv_rows, conv_cols);
            return;
        }

        //*------------------------------------ CALCULA O TAMANHO DA MATRIZ DE POOLING --------------------------------------------------
        int pool_rows = (int)((conv_rows - pooling_rows) / stride_pool_rows) + 1;
        int pool_cols = (int)((conv_cols - pooling_cols) / stride_pool_cols) + 1;
       
        //---------------------------------- ARMAZENA AS DIMENSÕES DA MATRIZ DE POOLING NO ARRAY POOLING_MATRIZ_SIZE (MATRIZ DE POOLING DE UMA DADA CAMADA É A MATRIZ DE ENTRADA DA CAMADA SEGUINTE) ------------------------------
         x->pooling_matriz_size[i* 2] = pool_rows; // Linhas da matriz de pooling
         x->pooling_matriz_size[i* 2 + 1] = pool_cols; // Colunas da matriz de pooling
        
        //-------------------------------- CALCULA O TAMANHO DA MATRIZ DE ENTRADA DE CADA CAMADA COM PADDING ------------------------------------------------
        //calcular o tamanho da matriz de entrada da camada atual (matriz de dados + padding) quando a camada atual for a primeira
        //calcular o tamanho da matriz de entrada da camada atual (matriz de pooling da camada anterior + padding) quando a camada atual não for a primeira
        int input_padded_rows = (i == 0) ? (int)x->input_size[0] + (2 * padding) : (int)x->pooling_matriz_size[(i - 1) * 2] + (2 * padding); //número de linhas da matriz de entrada
        int input_padded_cols = (i == 0) ? (int)x->input_size[1] + (2 * padding): (int)x->pooling_matriz_size[(i - 1) * 2 + 1] + (2 * padding); //número de colunas da matriz de entrada
        
       
        // --------------------------------- ARMAZENA AS DIMENSÕES DA MATRIZ DE ENTRADA DE CADA CAMADA NO ARRAY INPUT_PADDED ---------------------------------------------
        //se a camada atual for a primeira, armazena o tamanho da matriz de matriz de dados + padding
        //se a camada atual não for a primeira, armazena o tamanho da matriz de pooling da camada anterior + padding (matriz de entrada das camadas seguites à primeira é a matriz de pooling da camada anterior)
        x->input_padded[i*2] = input_padded_rows; //armazena o número de linhas da matriz de entrada + o padding 
        x->input_padded[i*2+1] = input_padded_cols; //armazena o número de colunas da matriz de entrada + o padding

        // -------------------------------- VERIFICA SE AS DIMENSÕES DO KERNEL EXCEDEM AS DIMENSÕES DA MATRIZ DE ENTRADA DE CADA CAMADA -----------------------------------
        if (kernel_rows > input_rows || kernel_cols > input_cols) {
            error("Layer %d: kernel (%d x %d) must be smaller than input matrix (%d x %d)", i + 1, kernel_rows, kernel_cols, input_rows, input_cols);
            return;
        }

        //---------------------------------------- VERIFICA SE AS DIMENSÕES DA JANELA DE POOLING EXCEDEM AS DIMENSÕES DA MATRIZ DE CONVOLUÇÃO -------------------------------
        if (pooling_rows > conv_rows || pooling_cols > conv_cols) {
            error("Layer %d: pooling window (%d x %d) must be smaller than convolution matrix (%d x %d)", i + 1, pooling_rows, pooling_cols, conv_rows, conv_cols);
            return;
        }
    }
    //* --------------------------------- ALOCA MEMÓRIA PARA A MATRIZ DE ENTRADA DE CADA CAMADA ---------------------------------------------
    //aloca memória para matriz de entrada de cada camada considerando o padding (matriz de dados + matriz de pooling da camada anterior) uma matriz por camada
    if(!alocar_matrizes(x, &x->input, x->num_layers, x->input_padded)){
        error("Error allocating memory for the kernels matrices");
        free(x->input);
        return;
    }
    //* --------------------------------- PREENCHE AS MATRIZES DE ENTRADA DE CADA CAMADA COM 0 ----------------------------------------------
    // Inicialize o estado aleatório
    unsigned int state = (unsigned int)x->random_state; // Use o tempo atual como semente
    matriz_fill(x, x->input, x->num_layers, x->input_padded, 0, 0, &state, gensym("Encoder input matrix")); //preenche as matrizes de entrada com 0 para facilitar o padding

    //* --------------------------------- ALOCA MEMÓRIA PARA A MATRIZ DE CONVOLUÇÃO ----------------------------------------------
    if(!alocar_matrizes(x, &x->convolution, x->num_layers, x->convolv_matriz_size)){ //argumentos: matriz de convolução, nº de matrizes, vetor de tamanho da matriz de convolução
        error("Error allocating memory for the convolution matrices");
        return;
    }
    //* --------------------------------- ALOCA MEMÓRIA PARA A MATRIZ DE POOLING ----------------------------------------------
    if(!alocar_matrizes(x, &x->pooling, x->num_layers, x->pooling_matriz_size)){
        error("Error allocating memory for the pooling matrices");
        return;
    }

    //* --------------------------------- ALOCA MEMÓRIA PARA A MATRIZ DE DELTA DA SAÍDA DE CADA CAMADA DO ENCODER ----------------------------------------------
    if(!alocar_matrizes(x, &x->delta, x->num_layers, x->pooling_matriz_size)){
        error("Error allocating memory for the encoder delta matrices");
        return;
    }

    //* --------------------------------- ALOCA MEMÓRIA PARA A MATRIZ DE GRADIENTE DA CONVOLUÇÃO ----------------------------------------------
    if(!alocar_matrizes(x, &x->conv_grad, x->num_layers, x->convolv_matriz_size)){ //argumentos: matriz de gradiente convolução, nº de matrizes, vetor de tamanho da matriz de convolução
        error("Error allocating memory for the convolution gradient matrices");
        return;
    }
    
    //* --------------------------------- ALOCA MEMÓRIA PARA O BUFFER DE SAÍDA DO ESPAÇO LATENTE ----------------------------------------------
    if (x->latent_out != NULL) {    
        freebytes(x->latent_out, x->pooling_matriz_size[(x->num_layers-1)*2+1] * sizeof(t_atom)); // Libera a memória do buffer de saída do espaço latente (nº de colunas da última camada do encoder)
        x->latent_out = NULL;
    }
    //aloca memória para o buffer de saída do espaço latente com o tamanho do nº de colunas da última camada do encoder
    x->latent_out = (t_atom *)getbytes(x->pooling_matriz_size[(x->num_layers-1)*2+1] * sizeof(t_atom)); 
    if (x->latent_out == NULL) {
        error("Error allocating memory for the latent output buffer");
        return;
    }

    //* --------------------------------------------- MENSAGEM PARA VERIFICAÇÃO ----------------------------------------------
    //  ENCODER
    for (int i = 0; i < x->num_layers; i++) {
        post("ENCODER:\n Layer %d:\n padding: %d\n input padded matrix: %d x %d\n kernel: %d x %d\n conv stride: %d x %d\n convolution matrix: %d x %d\n pooling window: %d x %d\n pooling stride: %d x %d\n pooling matrix: %d x %d",
         i + 1, //camada
         (int)x->padding[i], //padding
         (int)x->input_padded[i * 2], (int)x->input_padded[i * 2 + 1], //dimensões das matrizes de entrada
         (int)x->kernels_size[i * 2], (int)x->kernels_size[i * 2 + 1], //dimensões dos kernels
         (int)x->stride_conv[i * 2], (int)x->stride_conv[i * 2 + 1],//dimensões do stride de convolução
         (int)x->convolv_matriz_size[i * 2], (int)x->convolv_matriz_size[i * 2 + 1], //dimensões das matrizes de convolução
         (int)x->pooling_size[i * 2], (int)x->pooling_size[i * 2 + 1], //dimensões da janela de pooling
         (int)x->stride_pool[i * 2], (int)x->stride_pool[i * 2 + 1], //dimensões do stride de pooling
         (int)x->pooling_matriz_size[i * 2], (int)x->pooling_matriz_size[i * 2 + 1]); //dimensões das matrizes de pooling
    }
}


//*---------------------------------- CALCULA DIMENSÕES DAS MATRIZES DE CADA CAMADA DO DECODER ---------------------------------------------------
static void matrix_decoder_size(t_cnn2d *x){
    //esta função é chamada após a definição dos parâmetros da rede (nº de camadas, dimensões dos dados de entrada, dimensões dos kernels, padding, strides, etc)
    if (x->num_Tlayers <= 0) { // Verifica se o número de camadas do encoder é válido
        error("Invalid number of decoder layers: must be greater than 0");
        return;
    }
    post("Decoder layers: %d", x->num_Tlayers);

    //*----------------------------------------- libera memória das matrizes do decoder ----------------------------------------------------------
    //ATENÇÃO: É IMPORTANTE LIBERAR MEMÓRIA DAS MATRIZES ANTES DE MUDAR AS DIMENSÕES PARA EVITAR VAZAMENTO DE MEMÓRIA
    //obs: as matrizes de kernel do decoder são liberadas e alocadas na função decoder_kernel_size porque só dependem do número de camadas e das dimensões dos kernels

    free_matrix(x, &x->Tinput, x->num_Tlayers, x->Tinput_padded); //MATRIZ DE ENTRADA
    free_matrix(x, &x->Tconv, x->num_Tlayers, x->Tconv_size); //MATRIZ DE CONVOLUÇÃO
    free_matrix(x, &x->Tdelta, x->num_Tlayers, x->Tconv_size); //MATRIZ DE DELTA

    //*--------------------------------- CALCULA AS DIMENSÕES DAS MATRIZES DA CONVOLUÇÃO TRANSPOSTA (DECODER) ----------------------------------------------
    for(int j = 0; j < x->num_Tlayers; j++){ //percorre as camadas do decoder
    
        //------------------------------- RECUPERA DIMENSÕES DA MATRIZ DE ENTRADA  -----------------------------------------------------------------
        //!ATENÇÃO!!! NA CONVOLUÇÃO TRANSPOSTA O PADDING É APLICADO NA MATRIZ DE SAÍDA, REMOVENDO AS BORDAS AO INVÉS DE ADICIONÁ-LAS COM ZERO
        //Se for a primeira camada do decoder, a matriz de entrada é a matriz de pooling da última camada do encoder, caso contrário, é a matriz de convolução transposta da camada anterior
        int Tinput_rows = (j == 0) ? (int)x->pooling_matriz_size[(x->num_layers-1)*2]: (int)x->Tconv_size[(j - 1) * 2]; //número de linhas da matriz de entrada do decoder
        int Tinput_cols = (j == 0) ? (int)x->pooling_matriz_size[(x->num_layers-1)*2+1] : (int)x->Tconv_size[(j - 1) * 2 + 1]; //número de colunas da matriz de entrada do decoder
        // post("Layer %d: Tconv input matrix - %d x %d", j +1, Tinput_rows, Tinput_cols);

        //*------------------------------------- RECUPERA PADDING DO DECODER ------------------------------------------------------
        int Tpadding = (int)x->Tpadding[j]; //padding das matrizes de entrada do decoder 
        // post("(depuração)Layer %d: Tpadding - %d", j +1, Tpadding);  

        //*----------------------------------- RECUPERA DIMENSÕES DOS KERNELS DO DECODER -------------------------------------------
        int Tkernel_rows = (int)x->Tkernels_size[j * 2]; //linhas dos kernels do decoder 
        int Tkernel_cols = (int)x->Tkernels_size[j * 2 + 1]; //colunas dos kernels do decoder
        // post("(depuração)Layer %d: Tkernel matrix - %d x %d", j +1, Tkernel_rows, Tkernel_cols);
        
        //*------------------------------ RECUPERA DIMENSÕES DOS STRIDES DA CONVOLUÇÃO DO DECODER -----------------------------------
        int Tstride_conv_rows = (int)x->Tstride_conv[j * 2]; //stride da linha (convolução)
        int Tstride_conv_cols = (int)x->Tstride_conv[j * 2 + 1]; //stride da coluna (convolução)

        //--------------------------- verifica se os valores de stride são válidos -----------------------------------
        if (Tstride_conv_rows > Tkernel_rows || Tstride_conv_cols > Tkernel_cols) {
            error("Layer %d: transpose convolution stride must be less than or equal transpose cconvolution kernel", j + 1);
            return;
        }
        // post("(depuração)Layer %d: Tstride conv - %d x %d", j +1, Tstride_conv_rows, Tstride_conv_cols);

        //* -------------------------------- ARMAZENA AS DIMENSÕES DA MATRIZ DE ENTRADA DE CADA CAMADA DO DECODER NO ARRAY TINPUT_PADDED ---------------------------------------------
        //1º camada do decoder: matriz de pooling da última camada do encoder, demais camadas: matriz de convolução transposta da camada anterior
        x->Tinput_padded[j * 2] = Tinput_rows; //número de linhas da matriz de entrada do decoder
        x->Tinput_padded[j * 2 + 1] = Tinput_cols; //número de colunas da matriz de entrada do decoder 
        // post("(depuração)Layer %d: Tconv input padded matrix - %d x %d", j +1, Tinput_padded_rows, Tinput_padded_cols);

        //*----------------------------- CALCULA O TAMANHO DA MATRIZ DE CONVOLUÇÃO TRANSPOSTA DE CADA CAMADA DO DECODER --------------------------------------------------------
        int Tconv_rows = Tstride_conv_rows * (Tinput_rows - 1) + Tkernel_rows - 2 * Tpadding; //calcula o nº de linhas da matriz de convolução transposta considerando o padding
        int Tconv_cols = Tstride_conv_cols * (Tinput_cols -1) + Tkernel_cols - 2 * Tpadding; //calcula o nº de colunas da matriz de convolução transposta considerando o padding

        if(Tconv_rows <= 0 || Tconv_cols <= 0){ //verifica se as dimensões da matriz de convolução transposta são válidas
            error("Layer %d: invalid dimensions for transpose convolution matrix", j + 1);
            return;
        }
        
        //*---------------------------------- ARMAZENA AS DIMENSÕES DA MATRIZ DE CONVOLUÇÃO TRANSPOSTA DE CADA CAMADA NO ARRAY TCONV_SIZE ---------------------------------------------
        x->Tconv_size[j * 2] = Tconv_rows; //linhas da matriz de convolução transposta do decoder
        x->Tconv_size[j * 2 + 1] = Tconv_cols; //colunas da matriz de convolução transposta do decoder
    }
    //* --------------------------------- ALOCA MEMÓRIA PARA A MATRIZ DE ENTRADA DA CONVOLUÇÃO TRANSPOSTA DO DECODER ----------------------------------------------
    if(!alocar_matrizes(x, &x->Tinput, x->num_Tlayers, x->Tinput_padded)){ //argumentos: matriz de entrada da convolução transposta considerando padding, nº de matrizes, vetor de tamanho da matriz de entrada da convolução transposta
        error("Error allocating memory for the transposed convolution input matrices");
        return;
    }
    //*--------------------------------- ALOCA MEMÓRIA PARA A MATRIZ DE CONVOLUÇÃO TRANSPOSTA DO DECODER ----------------------------------------------
    if(!alocar_matrizes(x, &x->Tconv, x->num_Tlayers, x->Tconv_size)){ //argumentos: matriz de convolução transposta, nº de matrizes, vetor de tamanho da matriz de convolução transposta
        error("Error allocating memory for the transposed convolution matrices");
        return;
    }

    //* --------------------------------- ALOCA MEMÓRIA PARA A MATRIZ DE DELTA DO DECODER ----------------------------------------------
    if(!alocar_matrizes(x, &x->Tdelta, x->num_Tlayers, x->Tconv_size)){ //argumentos: matriz de delta, nº de matrizes, vetor de tamanho da matriz de entrada da convolução transposta
        error("Error allocating memory for the decoder delta matrices");
        return;
    }

    //* --------------------------------- ALOCA MEMÓRIA PARA O BUFFER DE SAÍDA DA MATRIZ RECONSTRUIDA ----------------------------------------------
    if (x->matrix_out != NULL) {    
        freebytes(x->matrix_out, x->Tconv_size[(x->num_Tlayers-1)*2+1] * sizeof(t_atom)); // Libera a memória do buffer de saída do espaço latente (nº de colunas da última camada do encoder)
        x->matrix_out = NULL;
    }
    //aloca memória para o buffer de saída do espaço latente com o tamanho do nº de colunas da matriz de saída do decoder
    x->matrix_out = (t_atom *)getbytes(x->Tconv_size[(x->num_Tlayers-1)*2+1] * sizeof(t_atom)); 
    if (x->matrix_out == NULL) {
        error("Error allocating memory for the matrix output buffer");
        return;
    }
   

    //*--------------------------------------------------------- MENSAGEM PARA VERIFICAÇÃO ---------------------------------------------------------------
    for (int j = 0; j < x->num_Tlayers; j++) {
        post("DECODER:\n Layer %d:\n padding: %d\n input matrix: %d x %d\n kernel: %d x %d\n conv stride: %d x %d\n transpose convolution matrix: %d x %d",
         j + 1, //camada
         (int)x->Tpadding[j], //padding
         (int)x->Tinput_padded[j * 2], (int)x->Tinput_padded[j * 2 + 1], //dimensões das matrizes de entrada
         (int)x->Tkernels_size[j * 2], (int)x->Tkernels_size[j * 2 + 1], //dimensões dos kernels
         (int)x->Tstride_conv[j * 2], (int)x->Tstride_conv[j * 2 + 1],//dimensões do stride de convolução
         (int)x->Tconv_size[j * 2], (int)x->Tconv_size[j * 2 + 1]); //dimensões das matrizes de convolução
    }
}

//*----------------------------------------- CRIA A REDE NEURAL -----------------------------------------------------
static void create_network(t_cnn2d *x){
    //* 1. verifica se o nº de camadas é válido
    if(x->num_layers <= 0 || x->num_Tlayers <= 0){
        error("Invalid number of layers: Please provide positive integer values for the number of layers");
        return;
    }
    //*2. calcula as dimensões das matrizes do encoder e aloca memória
    int total_layer = x->num_layers + x->num_Tlayers; //nº total de camadas (encoder + decoder)
    //calcula as dimensões das matrizes do encoder e aloca memória
    matrix_encoder_size(x);
    //calcula as dimensões das matrizes do decoder e aloca memória
    matrix_decoder_size(x);

    //* 3. aloca memória para vetor de bias
    free_bias(x, &x->bias_kernel); //libera memória do vetor de bias
    //aloca memória para o vetor de bias
    if(!alocar_bias(x, &x->bias_kernel, total_layer)){ //argumentos: estrutura, nº de camadas do encoder + nº de camadas do decoder
        error("Error allocating memory for the bias vector");
        return;
    }
    //preenche o vetor de bias com valores aleatórios
    unsigned int state = (unsigned int)x->random_state; // Use o tempo atual como semente
    bias_fill(x, x->bias_kernel, total_layer, 0, 0.3, &state); //preenche o vetor de bias com valores aleatórios entre 0 e 0.01

    //* 4. aloca memória para os momentos dos bias
    //obs: os momentos dos kernels são alocados e preenchidos nas funções kernel_size e decoder_kernel_size juntos com as matrizes de kernels
    //ATENÇÃO: os momentos dos bias são preenchidos com 0 automaticamente usando a função calloc (aloca e inicializa com 0)
    //primeiro momento do bias
    free_momtento_bias(x, &x->m_bias); //libera memória do vetor de momentos do bias
    if(!alocar_momento_bias(x, &x->m_bias, total_layer)){ //argumentos: estrutura, vetor de momentos do bias e tamanho 
        error("Error allocating memory for the bias momentum vector");
        return;
    }
    //segundo momento do bias
    free_momtento_bias(x, &x->v_bias); //libera memória do vetor de momentos do bias
    if(!alocar_momento_bias(x, &x->v_bias, total_layer)){ //argumentos: vetor de momentos do bias e tamanho 
        error("Error allocating memory for the bias momentum vector");
        return;
    }
    
    //* 5. inicializa os hiperparâmetros do otimizador Adam
    x->beta1 = 0.9; //beta1
    x->beta2 = 0.999; //beta2
    x->epsilon = 1e-8; //epsilon

    //* 6. inicializa contadores de exemplos e épocas
    x->current_data = 0; //inicializa o índice do vetor de dados
    x->current_epoch = 0; //inicializa o índice da época

    //* 7. verifica o modo de simentria da rede
    if (x->simetria == 1) { //se simetria = 1, cria a rede neural simétrica
        post("Symmetric convolutional autoencoder neural network (caenn2d) created");
    }
    else if (x->simetria == 0) { //se simetria = 0, cria a rede neural não simétrica
        post("Asymmetric convolutional autoencoder neural network (caenn2d) created");
    }

    //* 8. verifica se a matriz de convolução transposta da última camada do decoder tem o mesmo tamanho da matriz de dados de entrada
    //obs: a matriz de saída do decoder deve ter o mesmo tamanho da matriz de dados para a rede aprender a reconstruir a matriz de dados
    if(x->input_size[0] != x->Tconv_size[(x->num_Tlayers - 1) * 2] || x->input_size[1] != x->Tconv_size[(x->num_Tlayers - 1) * 2 + 1]){
        error("Input data matrix size does not match the size of the output matrix: Please adjust the network dimensions");
        return;
    }
}


//*---------------------------------- RECEBE AS DIMENSÕES DA MATRIZ DE ENTRADA -----------------------------------------
static void input_data_size(t_cnn2d *x, t_symbol *s, int argc, t_atom *argv){
    //* 1. recebe o nº de linhas e colunas da matriz de entrada
    //1.1 verifica se o argumento recebdio tem tamanho 2 (linha e coluna) -- sempre será 2
    if(argc != 2){ 
        error("Please provide number of rows and colunns for data matrix");
        return;
    }
    //*2. copia nº de linhas e colunas recebidos para o array input_size (vetor de dimensões da matriz de entrada)
    //2.1. verifica se os valores são maiores que zero
    for(int i = 0; i < argc; i++){
        if (argv[i].a_w.w_float <= 0) { 
            error("Input matriz dimensions must be integer and positive");
            return;
        }
        //*2.2. atribui as dimensões da matriz de entrada sem o padding (padding = 0)   
        x->input_size[i] = (int)argv[i].a_w.w_float; 
    }
    post("Input matriz: %d x %d (data_size)", (int)x->input_size[0], (int)x->input_size[1]);
}

//------------------------------------ determina o nº de camadas da cnn -------------------------------------------
static void num_camadas(t_cnn2d *x, t_symbol *s, int argc, t_atom *argv){ //recebe apenas um valor se a simetria = 1 (mesmo nº de camadas para encoder e decoder) e 2 valores se simetria = 0 (nº de camadas diferentes para encoder e decoder)    
    //libera memória das matrizes antes de atualizar o nº de camadas
    //obs: [IMPORTANTE] É IMPORTANTE LIBERAR A MEMÓRIA DAS MATRIZES ANTES DE ATUALIZAR O NÚMERO DE CAMADAS PARA EVITAR VAZAMENTO DE MEMÓRIA 
    //As matrizes só vão ser alocadas novamente após o cálculo das dimensões das matrizes de entrada, de convolução e de pooling na função create_network 
    //MATRIZES ENCODER   
    free_matrix(x, &x->input, x->num_layers, x->input_padded); //LIBERA MEMÓRIA DA MATRIZ DE ENTRADA
    free_matrix(x, &x->kernels, x->num_layers, x->kernels_size); //LIBERA MEMÓRIA DAS MATRIZES DE KERNEL
    free_matrix(x, &x->kernel_rotated, x->num_layers, x->kernels_size); //LIBERA MEMÓRIA DAS MATRIZES DE KERNEL
    free_matrix(x, &x->convolution, x->num_layers, x->convolv_matriz_size); // LIBERA MEMÓRIA DAS MATRIZES DE CONVOLUÇÃO
    free_matrix(x, &x->pooling, x->num_layers, x->pooling_matriz_size); // LIBERA MEMÓRIA DAS MATRIZES DE POOLING
    //matrizes de delta
    free_matrix(x, &x->delta, x->num_layers, x->pooling_matriz_size); // LIBERA MEMÓRIA DAS MATRIZES DE DELTA DA SAÍDA DE CADA CAMADA DO ENCODER
    free_matrix(x, &x->kernels_grad, x->num_layers, x->kernels_size); //LIBERA MEMÓRIA DAS MATRIZES DE GRADIENTE DO KERNEL
    free_matrix(x, &x->conv_grad, x->num_layers, x->convolv_matriz_size); // LIBERA MEMÓRIA DAS MATRIZES DE GRADIENTE CONVOLUÇÃO
    //matrizes do primeiro e segundo momento do kernel
    free_matrix(x, &x->m_kernel, x->num_layers, x->kernels_size); //LIBERA MEMÓRIA DAS MATRIZES DO PRIMEIRO MOMENTO DO KERNEL
    free_matrix(x, &x->v_kernel, x->num_layers, x->kernels_size); //LIBERA MEMÓRIA DAS MATRIZES DO SEGUNDO MOMENTO DO KERNEL

    //MATRIZES DECODER
    free_matrix(x, &x->Tinput, x->num_Tlayers, x->Tinput_padded); //LIBERA MEMÓRIA DA MATRIZ DE ENTRADA DO DECODER
    free_matrix(x, &x->Tkernel, x->num_Tlayers, x->Tkernels_size); //LIBERA MEMÓRIA DAS MATRIZES DE KERNEL DO DECODER
    free_matrix(x, &x->Tkernel_rotated, x->num_Tlayers, x->Tkernels_size); //LIBERA MEMÓRIA DAS MATRIZES DE KERNEL DO DECODER
    free_matrix(x, &x->Tconv, x->num_Tlayers, x->Tconv_size); // LIBERA MEMÓRIA DAS MATRIZES DE CONVOLUÇÃO DO DECODER
    //matriz de delta
    free_matrix(x, &x->Tdelta, x->num_Tlayers, x->Tconv_size); //LIBERA MEMÓRIA DA MATRIZ DE DELTA DO DECODER
    free_matrix(x, &x->Tkernel_grad, x->num_Tlayers, x->Tkernels_size); //LIBERA MEMÓRIA DAS MATRIZES DE GRADIENTE DO DECODER
    //matrizes do primeiro e segundo momento do kernel
    free_matrix(x, &x->m_Tkernel, x->num_Tlayers, x->Tkernels_size); //LIBERA MEMÓRIA DAS MATRIZES DE KERNEL DO DECODER
    free_matrix(x, &x->v_Tkernel, x->num_Tlayers, x->Tkernels_size); //LIBERA MEMÓRIA DAS MATRIZES DE KERNEL DO DECODER

    //*se simetria = ON
    if (x->simetria == 1) { //se simetria = 1, recebe apenas um valor para o nº de camadas
        if (argc != 1 || argv[0].a_type != A_FLOAT) { //verifica se o argumento é um float
            error("For symmetric mode, provide a single value for the number of encoder and decoder layers.");
            return;
        }
        x->num_layers = (int)argv[0].a_w.w_float; //atribui o nº de camadas
        x->num_Tlayers = x->num_layers; //nº de camadas do encoder = ao nº de camadas do decoder se simetria = 1
    }
    //*se simetria = OFF
    else if (x->simetria == 0){ //se simetria = 0, recebe dois valores para o nº de camadas (encoder e decoder)
        if (argc != 2 || argv[0].a_type != A_FLOAT || argv[1].a_type != A_FLOAT) { //verifica se os argumentos são floats
            error("For asymmetric mode, provide two positive values: number of encoder layers and number of decoder layers.");
            return;
        }
        x->num_layers = (int)argv[0].a_w.w_float; //nº de camadas do encoder
        x->num_Tlayers = (int)argv[1].a_w.w_float; //nº de camadas do decoder
    }
    
    //* aqui aloca memória pra Vetores com tamanhos relacionados ao número de camadas (função de ativação, método de pooling e dimensões de vetores)
    //obs: os vetores não precisam ser liberados antes de atualizar num_layers porque a função liberar_vetor não utiliza o valor de num_layers ou num_Tlayers

    //libera e aloca memória dos vetor de funções de ativação
    int total_layers = x->num_layers + x->num_Tlayers; //nº total de camadas = nº de camadas do encoder + nº de camadas do decoder
    if (x->activation_function_c != NULL) {
        freebytes(x->activation_function_c, total_layers * sizeof(t_symbol *)); // Libera a memória do array de nomes de funções de ativação
        x->activation_function_c = NULL;
    }
    x->activation_function_c = (t_symbol **)getbytes(total_layers * sizeof(t_symbol *)); // Aloca memória para o array de funções de ativação

    //libera e aloca memória do vetor de método de pooling
    if (x->pooling_function != NULL) {
        freebytes(x->pooling_function, x->num_layers * sizeof(t_symbol *)); // Libera a memória do array de métodos de pooling
        x->pooling_function = NULL;
    }
    // Aloca memória para o array de métodos de pooling com o tamanho do número de camadas do encoder (não precisa alocar para o decoder porque o pooling é feito somente no encoder)
    x->pooling_function = (t_symbol **)getbytes(x->num_layers * sizeof(t_symbol *)); 

    //libera e aloca memória dos vetor de dimensões e parâmetros

    //*ENCODER
    if (!ajustar_vetor(x, (t_float**)&x->padding, x->num_layers, "Error allocating memory for the encoder padding vector") || //vetor de tamanho do padding
        !ajustar_vetor(x, (t_float**)&x->input_padded, x->num_layers * 2, "Error allocating memory for the encoder input padded matrices size vector") || //vetor de tamanho da matriz de entrada com padding
        !ajustar_vetor(x, (t_float**)&x->kernels_size, x->num_layers * 2, "Error allocating memory for the encoder kernels size vector") || //vetor de tamanho dos kernels
        !ajustar_vetor(x, (t_float**)&x->convolv_matriz_size, x->num_layers * 2, "Error allocating memory for the encoder convolution matrices size vector") || //vetor de tamanho das matrizes de convolução
        !ajustar_vetor(x, (t_float**)&x->pooling_size, x->num_layers * 2, "Error allocating memory for the encoder pooling window size vector") || //vetor de tamanho da janela de pooling
        !ajustar_vetor(x, (t_float**)&x->stride_conv, x->num_layers * 2, "Error allocating memory for the encoder convolution stride size vector") || //vetor de tamanho do stride de convolução
        !ajustar_vetor(x, (t_float**)&x->stride_pool, x->num_layers * 2, "Error allocating memory for the encoder pooling stride size vector") || //vetor de tamanho do stride de pooling
        !ajustar_vetor(x, (t_float**)&x->pooling_matriz_size, x->num_layers * 2, "Error allocating memory for encoder the pooling matrices size vector")) { //vetor de tamanho das matrizes de pooling
        return; // Interrompe se houver erro
    }

    //*DECODER
    if (!ajustar_vetor(x, (t_float**)&x->Tpadding, x->num_Tlayers, "Error allocating memory for the decoder padding vector") || //vetor de tamanho do padding do decoder
        !ajustar_vetor(x, (t_float**)&x->Tinput_padded, x->num_Tlayers * 2, "Error allocating memory for the decoder input padded matrices size vector") || //vetor de tamanho da matriz de entrada do decoder com padding
        !ajustar_vetor(x, (t_float**)&x->Tkernels_size, x->num_Tlayers * 2, "Error allocating memory for the decoder kernels size vector") || //vetor de tamanho dos kernels do decoder
        !ajustar_vetor(x, (t_float**)&x->Tconv_size, x->num_Tlayers * 2, "Error allocating memory for the decoder convolution matrices size vector") || //vetor de tamanho das matrizes de convolução do decoder
        !ajustar_vetor(x, (t_float**)&x->Tstride_conv, x->num_Tlayers * 2, "Error allocating memory for the decoder convolution stride size vector")) { //vetor de tamanho do stride de convolução do decoder
        return; // Interrompe se houver erro
    }   

    //* verifica o modo de simetria da rede
    if (x->simetria == 1){
        post("Encoder and decoder layers: %d", x->num_layers);
    }

    if (x->simetria == 0){
        post("Encoder layers: %d, decoder layers: %d", x->num_layers, x->num_Tlayers);
    }
}

//*---------------------------------- RECEBE VALOR DE PADDING DAS CAMADAS DO ENCODER -------------------------------------------
static void padding(t_cnn2d *x, t_symbol *s, int argc, t_atom *argv){
    if(argc < x->num_layers){ //verifica se o nº de argumentos é compatível com o nº de camadas 
        error("Please provide a padding value for each layer");
        return;
    }
    //copia o valor de padding para cada camada
    for(int i = 0; i < argc; i++){
        x->padding[i] = argv[i].a_w.w_float; //atribui o valor de padding para cada camada
    }
    //se simetria = 1, os valores de padding são iguais para o encoder e o decoder
    if (x->simetria == 1){
        for(int i = 0; i < x->num_layers; i++){
            x->Tpadding[i] = x->padding[i]; //copiar os valores de padding do encoder para o decoder
            post("Encoder layer %d: padding %d - Decoder layer %d: padding %d", i+1, (int)x->padding[i], (int)x->Tpadding[i]); //imprime o valor de padding
            
        }
    } else { //se simetria = 0, os valores de padding são diferentes para o encoder e o decoder
        //imprime o valor de padding para cada camada
        for(int i = 0; i < x->num_layers; i++){
            post("Encoder layer %d: padding %d", i+1, (int)x->padding[i]); //imprime o valor de padding
        }
    }
}

//*---------------------------------- RECEBE OS VALORES DE PADDING DO DECODER ---------------------------------------------
static void decoder_padding(t_cnn2d *x, t_symbol *s, int argc, t_atom *argv){
    //verifica se simetria está ON
    if (x->simetria == 1) { //se simetria = 1, os valores de padding são iguais aos do encoder
        error("Decoder padding is only available for non-symmetric mode");
        return;
    }
    //se simetria = 0, recebe um valor de padding para cada matriz de entrada do decoder
    if (x->simetria == 0) { 
        if (argc != x->num_Tlayers) { //verifica se a lista de padding recebida tem o mesmo tamanho do nº de camadas do decoder
            error("Please provide a padding value for each decoder layer");
            return;
        }
        // Copia o valor de padding de cada camada do decoder para o vetor Tpadding
        for (int i = 0; i < argc; i++) {
            if (argv[i].a_w.w_float < 0) { // Verifica se o valor é um float e não negativo
                error("Decoder padding must be positive integers");
                return;
            }
            x->Tpadding[i] = argv[i].a_w.w_float; // Atribui o valor de padding recebido para cada camada do decoder
        }
    } 
    //imprime os valores de padding para cada camada do decoder
    for(int i = 0; i < x->num_Tlayers; i++){
        post("Decoder layer %d: padding: %d", i +1, (int)x->Tpadding[i]); //imprime o valor de padding
    }
}

//*---------------------------------- RECEBE AS DIMENSÕES DOS KERNELS DO ENCODER ---------------------------------------------
static void kernels_size(t_cnn2d *x, t_symbol *s, int argc, t_atom *argv) {
    //recebe lista com as dimensões dos kernels
    // Verifica se o número de elementos da lista é compatível com o nº de camadas (deve ser = 2 * nº de camadas)
    if (argc != x->num_layers * 2) {
        error("Please provide numbers of rows and columns for each encoder kernel");
        return;
    }
    //*-------------------------- LIBERA MEMÓRIA DAS MATRIZES DE KERNEL DO ENCODER ---------------------------------------------
    //!ATENÇÃO: É IMPORTANTE LIBERAR A MEMÓRIA DAS MATRIZES DE KERNEL ANTES DE ATUALIZAR OS VALORES DE X->KERNELS_SIZE PARA EVITAR VAZAMENTO DE MEMÓRIA  
    free_matrix(x, &x->kernels, x->num_layers, x->kernels_size); //argumentos: matriz de entrada, nº de matrizes, vetor de tamanho da matriz de entrada
    free_matrix(x, &x->kernel_rotated, x->num_layers, x->kernels_size); //kernel rotacionado
    free_matrix(x, &x->kernels_grad, x->num_layers, x->kernels_size); //libera memória das matrizes de gradiente do kernel antes de atualizar as dimensões
    free_matrix(x, &x->m_kernel, x->num_layers, x->kernels_size); //matrizes do primeiro momento do kernel
    free_matrix(x, &x->v_kernel, x->num_layers, x->kernels_size); //matrizes do segundo momento do kernel

    // Copia os pares de dimensões dos kernels para o array kernels_size
    for (int i = 0; i < argc; i++) {
        if (argv[i].a_w.w_float <= 0 || argv[i].a_w.w_float != (int)argv[i].a_w.w_float) { // Verifica se os valores são positivos e inteiros
            error("Encoder kernel dimensions must be integer and positive");
            return;
        }
        x->kernels_size[i] = (int)argv[i].a_w.w_float; // Atribui os valores ao vetor kernels_size
    }

    //* Se simetria = 1, os valores de kernel do encoder são iguais aos do decoder
    if(x->simetria == 1){
        //*libera memória das matrizes de kernel do decoder antes de atualizar as dimensões
        free_matrix(x, &x->Tkernel, x->num_Tlayers, x->Tkernels_size); //libera memória das matrizes de kernel do decoder antes de atualizar as dimensões
        free_matrix(x, &x->Tkernel_rotated, x->num_Tlayers, x->Tkernels_size); //libera memória das matrizes de kernel do decoder antes de atualizar as dimensões
        free_matrix(x, &x->Tkernel_grad, x->num_Tlayers, x->Tkernels_size); //libera memória das matrizes de gradiente do kernel do decoder antes de atualizar as dimensões
        free_matrix(x, &x->m_Tkernel, x->num_Tlayers, x->Tkernels_size); //libera memória das matrizes do primeiro momento do kernel do decoder antes de atualizar as dimensões
        free_matrix(x, &x->v_Tkernel, x->num_Tlayers, x->Tkernels_size); //libera memória das matrizes do segundo momento do kernel do decoder antes de atualizar as dimensões

        for(int i = 0; i < x->num_layers * 2; i++){ //copia os valores de kernel do decoder para o encoder
            x->Tkernels_size[i] = x->kernels_size[i];
        }
        //*----------------------------- ALOCA MEMÓRIA PARA AS MATRIZES DE KERNEL DO DECODER ---------------------------------------------
        //aloca memória para as matrizes de kernel do decoder com as novas dimensões
        if(!alocar_matrizes(x, &x->Tkernel, x->num_Tlayers, x->Tkernels_size)){
            error("Error allocating memory for the decoder kernels matrices");
            return;
        }
        if(!alocar_matrizes(x, &x->Tkernel_rotated, x->num_Tlayers, x->Tkernels_size)){
            error("Error allocating memory for the decoder kernels matrices");
            return;
        }
        //aloca memória para as matrizes de gradiente do kernel do decoder com as novas dimensões
        if(!alocar_matrizes(x, &x->Tkernel_grad, x->num_Tlayers, x->Tkernels_size)){
            error("Error allocating memory for the decoder gradient matrices");
            return;
        }
        //aloca memória para as matrizes do primeiro momento do kernel do decoder com as novas dimensões
        if(!alocar_matrizes(x, &x->m_Tkernel, x->num_Tlayers, x->Tkernels_size)){
            error("Error allocating memory for the decoder first moment kernels matrices");
            return;
        }
        //aloca memória para as matrizes do segundo momento do kernel do decoder com as novas dimensões
        if(!alocar_matrizes(x, &x->v_Tkernel, x->num_Tlayers, x->Tkernels_size)){
            error("Error allocating memory for the decoder second moment kernels matrices");
            return;
        }
        //*----------------------------- PREENCHE AS MATRIZES DE KERNEL DO DECODER ---------------------------------------------
        unsigned int state = (unsigned int)x->random_state; // Use o tempo atual como semente
        //Preenche as matrizes de kernel do decoder com valores aleatórios
        matriz_fill(x, x->Tkernel, x->num_Tlayers, x->Tkernels_size, 0, 1, &state, gensym("Decoder kernel"));
        //Preenche as matrizes do primeiro momento do kernel do decoder com zeros
        matriz_fill(x, x->m_Tkernel, x->num_Tlayers, x->Tkernels_size, 0, 0, &state, gensym("Decoder first moment kernel"));
        //Preenche as matrizes do segundo momento do kernel do decoder com zeros
        matriz_fill(x, x->v_Tkernel, x->num_Tlayers, x->Tkernels_size, 0, 0, &state, gensym("Decoder second moment kernel")); 

        for(int i = 0; i < x->num_layers; i++){ // Imprime as dimensões dos kernels do encoder
            post("Encoder layer: %d Kernel %d x %d - Decoder layer %d: Kernel %d x %d", i + 1, (int)x->kernels_size[i*2], (int)x->kernels_size[i*2+1], 
                                                                                                (int)x->Tkernels_size[i*2], (int)x->Tkernels_size[i*2+1]);
        }
    } else if (x->simetria == 0){ // Se simetria = 0, os valores de kernel do encoder são diferentes dos do decoder
        //imprime as dimensões dos kernels do encoder
        for(int i = 0; i < x->num_layers; i++){ // Imprime as dimensões dos kernels
            post("Encoder layer: %d Kernel %d x %d", i + 1, (int)x->kernels_size[i*2], (int)x->kernels_size[i*2+1]);
        }
    }
    //*----------------------------- ALOCA MEMÓRIA PARA AS MATRIZES DE KERNEL DO ENCODER ---------------------------------------------
    // Aloca memória para as matrizes de kernel com as novas dimensões
    if(!alocar_matrizes(x, &x->kernels, x->num_layers, x->kernels_size)){
        error("Error allocating memory for the encoder kernels matrices");
        return;
    }
    // Aloca memória para as matrizes de kernel rotacionadas
    if(!alocar_matrizes(x, &x->kernel_rotated, x->num_layers, x->kernels_size)){
        error("Error allocating memory for the encoder kernels matrices");
        return;
    }
    // Aloca memória para as matrizes de gradiente do kernel do encoder
    if(!alocar_matrizes(x, &x->kernels_grad, x->num_layers, x->kernels_size)){
        error("Error allocating memory for the encoder gradient kernels matrices");
        return;
    }
    // Aloca memória para as matrizes do primeiro momento do kernel
    if(!alocar_matrizes(x, &x->m_kernel, x->num_layers, x->kernels_size)){
        error("Error allocating memory for the encoder first moment kernels matrices");
        return;
    }
    // Aloca memória para as matrizes do segundo momento do kernel
    if(!alocar_matrizes(x, &x->v_kernel, x->num_layers, x->kernels_size)){
        error("Error allocating memory for the encoder second moment kernels matrices");
        return;
    }
    //*----------------------------- PREENCHE AS MATRIZES DE KERNEL DO ENCODER ---------------------------------------------
    unsigned int state = (unsigned int)x->random_state; // Use o tempo atual como semente
    // Preenche as matrizes de kernel com valores aleatórios
    matriz_fill(x, x->kernels, x->num_layers, x->kernels_size, 0, 1, &state, gensym("Encoder kernel"));
    //preeche as matrizes do primeiro momento do kernel com zeros
    matriz_fill(x, x->m_kernel, x->num_layers, x->kernels_size, 0, 0, &state, gensym("Encoder first moment kernel"));
    //preeche as matrizes do segundo do kernel com zeros
    matriz_fill(x, x->v_kernel, x->num_layers, x->kernels_size, 0, 0, &state, gensym("Encoder second moment kernel"));
 
}

//*---------------------------------- RECEBE AS DIMENSÕES DOS KERNELS DO DECODER ---------------------------------------------
static void decoder_kernel_size(t_cnn2d *x, t_symbol *s, int argc, t_atom *argv) {
    //verifica se simetria está ON
    if (x->simetria == 1) { //se simetria = 1, os valores de padding são iguais aos do encoder
        error("Decoder kernels dimensions are only available for non-symmetric mode");
        return;
    }
    if(argc != x->num_Tlayers * 2){ //verifica se o nº de argumentos é compatível com o nº de camadas do decoder
        error("Please provide numbers of rows and columns for each decoder kernel");
        return;
    }
    // Se simetria = 0, recebe um valor de linha e coluna do kernel de cada camada do decoder
    if (x->simetria == 0){
        //libera memória das matrizes de kernel do decoder antes de atualizar as dimensões
        free_matrix(x, &x->Tkernel, x->num_Tlayers, x->Tkernels_size); 
        free_matrix(x, &x->Tkernel_rotated, x->num_Tlayers, x->Tkernels_size); 
        free_matrix(x, &x->Tkernel_grad, x->num_Tlayers, x->Tkernels_size); //libera memória das matrizes de gradiente do kernel do decoder antes de atualizar as dimensões
        free_matrix(x, &x->m_Tkernel, x->num_Tlayers, x->Tkernels_size); //libera memória das matrizes do primeiro momento do kernel do decoder antes de atualizar as dimensões
        free_matrix(x, &x->v_Tkernel, x->num_Tlayers, x->Tkernels_size); //libera memória das matrizes do segundo momento do kernel do decoder antes de atualizar as dimensões

        for (int i = 0; i < argc; i++) { //percorre a lista de argumentos recebida 
            if (argv[i].a_w.w_float <= 0 || argv[i].a_w.w_float != (int)argv[i].a_w.w_float) { // Verifica se os valores são positivos e inteiros
                error("Decoder kernel dimensions must be integer and positive");
                return;
            }
            x->Tkernels_size[i] = (int)argv[i].a_w.w_float; // Atribui os valores recebidos ao vetor Tkernels_size
        }
    } 
    //imprime as dimensões dos kernels do decoder
    for(int k = 0; k < x->num_Tlayers; k++){ // Imprime as dimensões dos kernels
        post("Decoder layer: %d Kernel %d x %d", k + 1, (int)x->Tkernels_size[k*2], (int)x->Tkernels_size[k*2+1]);
    }

    //*----------------------------- ALOCA MEMÓRIA PARA AS MATRIZES DE KERNEL DO DECODER ---------------------------------------------
    // Aloca memória para as matrizes de kernel com as novas dimensões
    if(!alocar_matrizes(x, &x->Tkernel, x->num_Tlayers, x->Tkernels_size)){
        error("Error allocating memory for the decoder kernels matrices");
        return;
    }
     // Aloca memória para as matrizes de kernel com as novas dimensões
     if(!alocar_matrizes(x, &x->Tkernel_rotated, x->num_Tlayers, x->Tkernels_size)){
        error("Error allocating memory for the decoder kernels matrices");
        return;
    }
    // aloca memória para as matrizes de gradiente do kernel do decoder
    if(!alocar_matrizes(x, &x->Tkernel_grad, x->num_Tlayers, x->Tkernels_size)){
        error("Error allocating memory for the decoder gradient matrices");
        return;
    }
    // aloca memória para as matrizes do primeiro momento do kernel do decoder
    if(!alocar_matrizes(x, &x->m_Tkernel, x->num_Tlayers, x->Tkernels_size)){
        error("Error allocating memory for the decoder first moment kernels matrices");
        return;
    }
    // aloca memória para as matrizes do segundo momento do kernel do decoder
    if(!alocar_matrizes(x, &x->v_Tkernel, x->num_Tlayers, x->Tkernels_size)){
        error("Error allocating memory for the decoder second moment kernels matrices");
        return;
    }
    //*----------------------------- PREENCHE AS MATRIZES DE KERNEL DO DECODER ---------------------------------------------
    unsigned int state = (unsigned int)x->random_state; // Use o tempo atual como semente
    // Preenche as matrizes de kernel com valores aleatórios
    matriz_fill(x, x->Tkernel, x->num_Tlayers, x->Tkernels_size, 0, 1, &state, gensym("Decoder kernel")); //argumentos: objeto, matriz de kernel, nº de matrizes, vetor de tamanho da matriz de kernel, valor mínimo, valor máximo
    // Preenche as matrizes do primeiro momento do kernel com zeros
    matriz_fill(x, x->m_Tkernel, x->num_Tlayers, x->Tkernels_size, 0, 0, &state, gensym("Decoder first moment kernel"));
    // Preenche as matrizes do segundo momento do kernel com zeros
    matriz_fill(x, x->v_Tkernel, x->num_Tlayers, x->Tkernels_size, 0, 0, &state, gensym("Decoder second moment kernel"));
}


//*---------------------------------- RECEBE DIMENSÕES DA JANELA DE POOLING -----------------------------------------------
//! ATENÇÃO: ESSES VALORES NÃO SÃO DIMENSÕES DA MATRIZ DE POOLING, SÃO TAMANHOS DA JANELA DE OPERAÇÃO DE POOLING REALIZADA NA MATRIZ DE CONVOLUÇÃO
static void pooling_size(t_cnn2d *x, t_symbol *s, int argc, t_atom *argv) {
    // Verifica se o número de argumentos é compatível
    if (argc != x->num_layers * 2) {
        error("Please provide numbers of rows and columns for each pooling window");
        return;
    }
    // Copia os pares de dimensões dos poolings para o array pooling_size
    for (int i = 0; i < argc; i++) {
        // Verifica se os valores são positivos e inteiros
        if (argv[i].a_w.w_float <= 0 || argv[i].a_w.w_float != (int)argv[i].a_w.w_float) {
            error("Pooling dimensions must be integer and positive");
            return;
        }
        x->pooling_size[i] = (int)argv[i].a_w.w_float; // Copia os valores para o array pooling_size
    }
           
    for(int p = 0; p < x->num_layers; p++){
     // Imprime as dimensões da janela de pooling
        post("Layer: %d pooling window %d x %d", p + 1, (int)x->pooling_size[p*2], (int)x->pooling_size[p*2+1]);
    }
}

//*---------------------------------- RECEBE AS DIMENSÕES DOS STRIDES DA CONVOLUÇÃO DO ENCODER -------------------------------------------
static void stride_convolution(t_cnn2d *x, t_symbol *s, int argc, t_atom *argv) {
    // Verifica se o número de argumentos é compatível (pares de valores para cada camada)
    if (argc != x->num_layers * 2) {
        error("Please provide rows and columns for each convolution stride");
        return;
    }
    // Copia os valores de stride para cada camada
    for (int i = 0; i < argc; i++) {
        // Valida os valores de stride
        if (argv[i].a_w.w_float <= 0 || argv[i].a_w.w_float != (int)argv[i].a_w.w_float) {
            error("Convolution stride %d must be positive integers", i + 1);
            return;
        }
        x->stride_conv[i] = (int)argv[i].a_w.w_float;
    }
    //se simetria = 1, os valores de stride do encoder são iguais aos do decoder
    if (x->simetria == 1) {
        for (int i = 0; i < x->num_layers * 2; i++) {
            x->Tstride_conv[i] = x->stride_conv[i]; //copia os valores de stride do encoder para o decoder
        }
        // Imprime os valores de stride
        for (int i = 0; i < x->num_layers; i++) {
            post("Encoder layer: %d Convolution stride %d x %d - Decoder layer: %d Convolution stride %d x %d", i + 1, (int)x->stride_conv[i * 2], (int)x->stride_conv[i * 2 + 1], 
                                                                                                                i + 1, (int)x->Tstride_conv[i * 2], (int)x->Tstride_conv[i * 2 + 1]);
        }
    } else if (x->simetria == 0) { //se simetria = 0, os valores de stride são diferentes para o encoder e o decoder
        // Imprime os valores de stride do encoder apenas
        for (int i = 0; i < x->num_layers; i++) {
            post("Encoder layer: %d Convolution stride %d x %d", i + 1, (int)x->stride_conv[i * 2], (int)x->stride_conv[i * 2 + 1]);
        }
    }
}

//*--------------------------------- RECEBE AS DIMENSÕES DOS STRIDES DO DECODER -------------------------------------------
static void decoder_stride (t_cnn2d *x, t_symbol *s, int argc, t_atom *argv){
    //verifica se simetria está ON
    if (x->simetria == 1) { //se simetria = 1, os valores de padding são iguais aos do encoder
        error("Decoder convolution stride is only available for non-symmetric mode");
        return;
    }
    if(argc != x->num_Tlayers * 2){ //verifica se o nº de argumentos é compatível com o nº de camadas do decoder
        error("Please provide numbers of rows and columns for each decoder convolution stride");
        return;
    }
    //verifica se simetria está OFF
    if (x->simetria == 0) { //se simetria = 0, recebe pares de valores de stride para cada camada do decoder
        for (int i = 0; i < argc; i++) {
            // Verifica se os valores são positivos e inteiros
            if (argv[i].a_w.w_float <= 0 || argv[i].a_w.w_float != (int)argv[i].a_w.w_float) {
                error("Decoder convolution stride must be integer and positive");
                return;
            }
            x->Tstride_conv[i] = (int)argv[i].a_w.w_float; // Copia os valores para o array Tstride_conv
        }
    } 
    // Imprime os valores de stride da convolução do decoder
    for (int i = 0; i < x->num_Tlayers; i++) {
        post("Decoder layer: %d Decoder convolution stride %d x %d", i + 1, (int)x->Tstride_conv[i * 2], (int)x->Tstride_conv[i * 2 + 1]);
    }
}

//*---------------------------------- RECEBE AS DIMENSÕES DOS STRIDES DO POOLING -------------------------------------------
static void stride_pooling(t_cnn2d *x, t_symbol *s, int argc, t_atom *argv) {
    // Verifica se o número de argumentos é compatível
    if (argc != x->num_layers * 2) {
        error("Please provide rows and columns for each Pooling stride");
        return;
    }
    // Copia os valores de stride de pooling para cada camada
    for (int i = 0; i < argc; i++) {
        
        // Valida os valores de stride
       if (argv[i].a_w.w_float <= 0 || argv[i].a_w.w_float != (int)argv[i].a_w.w_float) {
            error("Layer %d: pooling stride must be positive integers", i + 1);
            return;
        }       
        // Armazena os valores no array `stride_pool`
        x->stride_pool[i] = (int)argv[i].a_w.w_float;
    }
    // Imprime os valores de stride
    for (int i = 0; i < x->num_layers; i++) {
        post("Layer: %d pooling stride %d x %d", i + 1, (int)x->stride_pool[i * 2], (int)x->stride_pool[i * 2 + 1]);
    }
}


//*------------------------------------------ CONVOLUÇÃO E POOLING 2D --------------------------------------------------
static void convolution_pooling(t_cnn2d *x){ //função está funcionando corretamente
    //* para cada camada do encoder, realiza a convolução (convolução -> soma do bias -> ativação) e o pooling (max ou avg)
    for(int m = 0; m < x->num_layers; m++){ //percorre cada matriz de entrada do encoder
        //* 1. recupera dimensões das matrizes (entrada, kernel, convolução) e parâmetros (stride, pooling)
        //*dimensões da matriz de entrada
        // int input_rows = x->input_padded[m*2]; //nº de linhas da matriz de entrada com padding
        // int input_cols = x->input_padded[m*2+1]; //nº de colunas da matriz de entrada com padding
        //* padding
        // int padding = x->padding[m]; //valor de padding para a matriz de entrada
        //*dimensões do kernel
        int kernel_rows = x->kernels_size[m*2]; //nº de linhas do kernel
        int kernel_cols = x->kernels_size[m*2+1]; // nº de colunas do kernel
        //*dimensões do stride
        int stride_rows = x->stride_conv[m*2]; //stride de convolução para as linhas
        int stride_cols = x->stride_conv[m*2+1]; //stride de convolução para as colunas
        //*dimensões da matriz de convolução
        int conv_rows = x->convolv_matriz_size[m*2]; //nº de linhas da matriz de convolução
        int conv_cols = x->convolv_matriz_size[m*2+1]; //nº de colunas da matriz de convolução

        //* 2. recupera a função de ativação para cada camada
        activation_func activation = get_activation_function(x->activation_function_c[m]);
        if(activation == NULL){ //verifica se a função de ativação é válida
            error("Encoder Layer %d: Invalid activation function", m+1);
            return;
        }
        
        //![DEPURAÇÃO
        // //[DEPURAÇÃO] imprime a função de ativação
        // post("Encoder Layer %d: Activation function: %s", m + 1, x->activation_function_c[m]->s_name);
        //(DEPURAÇÃO)imprime os kernels
        // post("Encoder Layer %d: Kernels", m + 1); //imprime o nº da camada atual
        // for(int i = 0; i < kernel_rows; i++){ //percorre as linhas do kernel
        //     for(int j = 0; j < kernel_cols; j++){ //percorre as colunas do kernel
        //         post("%f ", x->kernels[m][i][j]); //imprime o valor do kernel
        //     }
        //     post(""); //nova linha após cada linha do kernel
        // }
        //![DEPURAÇÃO]
        
        // //* 3. CONVOLUÇÃO    
        for(int i = 0; i < conv_rows; i++){ //percorre as linhas da matriz de convolução
            int in_i = i * stride_rows; //posição inicial da linha na matriz entrada de acordo com o stride
            for(int j = 0; j < conv_cols; j++){ //percorre as colunas da matriz de convolução
                int in_j = j * stride_cols; //posição inicial da coluna na matriz entrada de acordo com o stride
                float convol = 0.0; //inicializa a variável de soma com zero
                for(int k = 0; k < kernel_rows; k++){ //percorre as linhas do kernel
                    for(int l = 0; l < kernel_cols; l++){ //percorre as colunas do kernel
                        convol += x->input[m][in_i+k][in_j+l] * x->kernels[m][k][l]; //para cada região do kernle na matriz de entrada, realiza a convolução com o kernel
                    }
                }
                //cada valor da convolução deve ser somado ao bias da camada
                convol += x->bias_kernel[m]; //soma o resultado da convolução ao bias da camada

                //função de ativação
                x->convolution[m][i][j] = activation(convol); //atribui o valor da convolução à matriz de convolução após passar pela função de ativação
            }   
        }

        //![DEPURAÇÃO]
        // //((DEPURAÇÃO)imprime a matriz de convolução
        // post("Encoder Layer %d: Convolution matrix", m + 1); //imprime o nº da camada atual
        // for(int i = 0; i < conv_rows; i++){ //percorre as linhas da matriz de convolução
        //     for(int j = 0; j < conv_cols; j++){ //percorre as colunas da matriz de convolução
        //         post("%f ", x->convolution[m][i][j]); //imprime o valor da matriz de convolução
        //     }
        //     post(""); //nova linha após cada linha da matriz
        // }
        //![DEPURAÇÃO]

        // //* 4. POOLING
        //dimensões da janela de pooling
        int pooling_rows = x->pooling_size[m*2]; //recupera o nº de linhas da janela de pooling
        int pooling_cols = x->pooling_size[m*2+1]; //recupera o nº de colunas da janela de pooling

        //dimensões da matriz de pooling
        int pooling_matriz_rows = x->pooling_matriz_size[m*2]; //recupera o nº de linhas da matriz de pooling
        int pooling_matriz_cols = x->pooling_matriz_size[m*2+1]; //recupera o nº de colunas da matriz de pooling

        //dimensões do stride de pooling
        int stride_pool_rows = x->stride_pool[m*2]; //recupera o stride de pooling para as linhas
        int stride_pool_cols = x->stride_pool[m*2+1]; //recupera o stride de pooling para as colunas

        //recupera o método de pooling de cada camada
        pooling_func pooling = get_pooling_method(x->pooling_function[m]);

        // //[DEPURAÇÃO] imprime o método de pooling
        // post("Encoder Layer %d: Pooling function: %s", m + 1, x->pooling_function[m]->s_name);


        //calcula o pooling com o método escolhido para cada camada
        pooling(x->convolution[m], conv_rows, conv_cols, x->pooling[m], pooling_matriz_rows, pooling_matriz_cols, pooling_rows, pooling_cols, stride_pool_rows, stride_pool_cols); //realiza o pooling na matriz de convolução

        //![DEPURAÇÃO]
        // //(DEPURAÇÃO)imprime a matriz de pooling
        // post("Encoder Layer %d: Pooling matrix", m + 1); //imprime o nº da camada atual
        // for(int i = 0; i < pooling_matriz_rows; i++){ //percorre as linhas da matriz de pooling
        //     for(int j = 0; j < pooling_matriz_cols; j++){ //percorre as colunas da matriz de pooling
        //         post("%f ", x->pooling[m][i][j]); //imprime o valor da matriz de pooling
        //     }
        //     post(""); //nova linha após cada linha da matriz
        // }
        //![DEPURAÇÃO]


        // //* 5. SE A CAMADA ATUAL NÃO FOR A ÚLTIMA DO ENCODER: 
        // //* COPIA A MATRIZ DE POOLING PARA A MATRIZ DE ENTRADA DA PRÓXIMA CAMADA IGNORANDO A REGIÃO DE PADDING SE HOUVER
        //OBS: TODAS MATRIZES DE ENTRADA JÁ FORAM PREENCHDIDAS COM ZEROS NA FUNÇÃO MATRIX_ENCODER_SIZE PARA FACILITAR A CÓPIA MANTENDO AS REGIÕES DE PADDING
        if (m < x->num_layers - 1) { // Verifica se a camada atual não é a última camada
            int next_input_rows = x->input_padded[(m + 1) * 2]; // recupera o nº de linhas da matriz de entrada da próxima camada com padding se houver
            int next_input_cols = x->input_padded[(m + 1) * 2 + 1]; // recupera o nº de colunas da matriz de entrada da próxima camada com padding se houver

            //recupera o valor de padding da matriz de entrada da próxima camada
            int next_padding = x->padding[m + 1]; //recupera o valor de padding da matriz de entrada da próxima camada

            //verifica se o o tamanho da matriz de pooling da camada atual é compatível com tamanho da matriz de entrada da próxima camada sem o padding (devem ser iguais)
            if (next_input_rows - 2 * next_padding != pooling_matriz_rows || next_input_cols - 2 * next_padding != pooling_matriz_cols) {
                error("Pooling matrix of layer %d is not compatible with input matrix dimensions of layer %d", m + 1, m + 2);
                return;
            }
            
            //copia a matriz de pooling para a matriz de entrada da próxima camada ignorando a região de padding
            for (int k = next_padding; k < next_input_rows - next_padding; k++) { // percorre as linhas da matriz de entrada da próxima camada
                for (int l = next_padding; l < next_input_cols - next_padding; l++) { // percorre as colunas da matriz de entrada da próxima camada
                    x->input[m + 1][k][l] = x->pooling[m][k - next_padding][l - next_padding]; //copia os valores da matriz de pooling da camada atual para a matriz de entrada  da próxima camada ignorando regiões de padding
                }
            }

            //![DEPURAÇÃO]
            // //(DEPURAÇÃO)imprime a matriz de entrada da próxima camada
            // post("Encoder Layer %d: Input matrix", m + 2); //imprime o nº da camada atual
            // for (int i = 0; i < next_input_rows; i++) { //percorre as linhas da matriz de entrada da próxima camada
            //     for (int j = 0; j < next_input_cols; j++) { //percorre as colunas da matriz de entrada da próxima camada
            //         post("%f ", x->input[m + 1][i][j]); //imprime o valor da matriz de entrada da próxima camada
            //     }
            //     post(""); //nova linha após cada linha da matriz
            // }
            //![DEPURAÇÃO]
        }
        
        // //* 6. SE A CAMADA ATUAL FOR A ÚLTIMA DO ENCODER: 
        // //* COPIA A MATRIZ DE POOLING PARA A MATRIZ DE ENTRADA DA PRIMEIRA CAMADA DO DECODER E IMPRIME A MATRIZ DE POOLING DA ÚLTIMA CAMADA DO ENCODER (ESTA MATRIZ É O ESPAÇO LATENTE -- DADOS COMPRIMIDOS)
        else{ 
            //copia a matriz de pooling da última camada do encoder para a matriz de entrada da primeira camada do decoder
            for (int k = 0; k < x->Tinput_padded[0]; k++) { // percorre as linhas da matriz de entrada do decoder
                for (int l = 0; l < x->Tinput_padded[1]; l++) { // percorre as colunas da matriz de entrada do decoder
                    x->Tinput[0][k][l] = x->pooling[m][k][l]; //copia os valores da matriz de pooling da última camada do encoder para a matriz de entrada da primeira camada do decoder
                }
            }
            //![DEPURAÇÃO]
            // post("Encoder Layer %d: Latente space", m + 1); //imprime o nº da camada atual (última camada do encoder)
            // for (int i = 0; i < pooling_matriz_rows; i++) { //percorre as linhas da matriz de pooling
            //     for (int j = 0; j < pooling_matriz_cols; j++) { //percorre as colunas da matriz de pooling
            //         post("%f ", x->pooling[m][i][j]); //imprime o valor da matriz de pooling
            //     }
            //     post(""); // Nova linha após cada linha da matriz
            // }
            //![DEPURAÇÃO]

        }
    }
}



//*----------------------------------- CONVOLUÇÃO TRANSPOSTA -------------------------------------
static void transposed_convolution (t_cnn2d *x){ 
    for(int m = 0; m < x->num_Tlayers; m++){ //percorre cada matriz de entrada do decoder

        //* 1. dimensões das matrizes do decoder
        int Tinput_rows = x->Tinput_padded[m*2]; //recupera o nº de linhas da matriz de entrada do decoder com padding
        int Tinput_cols = x->Tinput_padded[m*2+1]; // recupera nº de colunas da matriz de entrada do decoder om padding
        //dimensões do kernel
        int Tkernel_rows = x->Tkernels_size[m*2]; //recupera nº de linhas do kernel
        int Tkernel_cols = x->Tkernels_size[m*2+1]; //recupera nº de colunas do kernel
        //dimensões do stride
        int Tconv_stride_rows = x->Tstride_conv[m*2]; //recupera o stride de convolução transposta para as linhas
        int Tconv_stride_cols = x->Tstride_conv[m*2+1]; //recupera o stride de convolução transposta para as colunas
        //dimensões da matriz de convolução
        int Tconv_rows = x->Tconv_size[m*2]; //recupera o nº de linhas da matriz de convolução transposta
        int Tconv_cols = x->Tconv_size[m*2+1]; //recupera o nº de colunas da matriz de convolução transposta

        int Tpadding = x->Tpadding[m]; //recupera o valor de padding de cada camada do decoder

        //* 2. atribui a função de ativação para cada camada
        activation_func activation = get_activation_function(x->activation_function_c[m+x->num_layers]); //recupera a função de ativação das camadas do decoder
        if(activation == NULL){ //verifica se a função de ativação é válida
            error("Decoder Layer %d: Invalid activation function", m+1);
            return;
        }

        // //[DEPURAÇÃO] imprime a função de ativação
        // post("Decoder Layer %d: Activation function: %s", m + 1, x->activation_function_c[m+x->num_layers]->s_name);

        //![DEPURAÇÃO] 
        // //imprime a matriz de entrada do decoder (DEPURAÇÃO)
        // post("Decoder Layer %d: Input matrix", m + 1); //imprime o nº da camada atual
        // for(int i = 0; i < Tinput_rows; i++){ //percorre as linhas da matriz de entrada do decoder
        //     for(int j = 0; j < Tinput_cols; j++){ //percorre as colunas da matriz de entrada do decoder
        //         post("%f ", x->Tinput[m][i][j]); //imprime o valor da matriz de entrada do decoder
        //     }
        //     post(""); //nova linha após cada linha da matriz
        // }

        // //(DEPURAÇÃO)imprime os kernels
        // post("Decoder Layer %d: Kernels", m + 1); //imprime o nº da camada atual
        // for(int i = 0; i < Tkernel_rows; i++){ //percorre as linhas do kernel
        //     for(int j = 0; j < Tkernel_cols; j++){ //percorre as colunas do kernel
        //         post("%f ", x->Tkernel[m][i][j]); //imprime o valor do kernel
        //     }
        //     post(""); //nova linha após cada linha do kernel
        // }
        //![DEPURAÇÃO]
        
        // //* 3. CONVOLUÇÃO TRANSPOSTA 
                // Inicializa a saída com zeros
        for (int i = 0; i < Tconv_rows; i++) {
            for (int j = 0; j < Tconv_cols; j++) {
                x->Tconv[m][i][j] = 0.0;
            }
        }
        // Percorre cada elemento da matriz de entrada do decoder
        for (int i_in = 0; i_in < Tinput_rows; i_in++) { //percorre as linhas da matriz de entrada do decoder
            for (int j_in = 0; j_in < Tinput_cols; j_in++) { //percorre as colunas da matriz de entrada do decoder
                // Posição inicial na saída (considerando stride)
                int i_out_center = i_in * Tconv_stride_rows; //posição inicial da linha na matriz de saída
                int j_out_center = j_in * Tconv_stride_cols; //posição inicial da coluna na matriz de saída
                // percorre o kernel
                for (int k = 0; k < Tkernel_rows; k++) {
                    for (int l = 0; l < Tkernel_cols; l++) {
                        // Posição na saída (subtrai padding)
                        int i_out = i_out_center + k - Tpadding;
                        int j_out = j_out_center + l - Tpadding;

                        // Verifica se a posição da matriz de saída é válida considerando o corte do padding
                        if (i_out >= 0 && i_out < Tconv_rows && j_out >= 0 && j_out < Tconv_cols) {
                            x->Tconv[m][i_out][j_out] += x->Tinput[m][i_in][j_in] * x->Tkernel[m][k][l]; //realiza a convolução transposta
                        }
                    }
                }
            }
        }

        // Adiciona o bias e aplica a função de ativação
        for (int i = 0; i < Tconv_rows; i++) {
            for (int j = 0; j < Tconv_cols; j++) {
                x->Tconv[m][i][j] = activation(x->Tconv[m][i][j] + x->bias_kernel[m + x->num_layers]);
            }
        }

        //![DEPURAÇÃO]
        // //imprime a matriz de convolução transposta
        // post("Decoder Layer %d: Transposed convolution matrix", m + 1); //imprime o nº da camada atual
        // for (int i = 0; i < Tconv_rows; i++) { //percorre as linhas da matriz de convolução transposta
        //     for (int j = 0; j < Tconv_cols; j++) { //percorre as colunas da matriz de convolução transposta
        //         post("%f ", x->Tconv[m][i][j]); //imprime o valor da matriz de convolução transposta
        //     }
        //     post(""); //nova linha após cada linha da matriz
        // }
        //![DEPURAÇÃO]

        //* 4. SE A CAMADA ATUAL NÃO FOR A ÚLTIMA: 
        //* COPIA A MATRIZ DE CONVOLUÇÃO TRANSPOSTA PARA A MATRIZ DE ENTRADA DA PRÓXIMA CAMADA DO DECODER
        if(m < x->num_Tlayers-1){
            //copia a matriz de convolução transposta para a matriz de entrada da próxima camada do decoder
            for (int i = 0; i < Tconv_rows; i++) { //percorre as linhas da matriz de convolução transposta
                for (int j = 0; j < Tconv_cols; j++) { //percorre as colunas da matriz de convolução transposta
                    x->Tinput[m + 1][i][j] = x->Tconv[m][i][j]; //copia os valores da matriz de convolução transposta para a matriz de entrada da próxima camada do decoder
                    
                }
            }
        }
        //* 5. SE A CAMADA ATUAL FOR A ÚLTIMA:
        //* IMPRIME A MATRIZ DE CONVOLUÇÃO TRANSPOSTA DA ÚLTIMA CAMADA DO DECODER (MATRIZ RECONSTRUÍDA)
        //![DEPURAÇÃO]
        // else{ 
        //     post("Decoder Layer %d: Reconstructed matrix", m + 1); //imprime o nº da camada atual
        //     for (int i = 0; i < Tconv_rows; i++) { //percorre as linhas da matriz de convolução transposta
        //         for (int j = 0; j < Tconv_cols; j++) { //percorre as colunas da matriz de convolução transposta
        //             post("%f ", x->Tconv[m][i][j]); //imprime o valor da matriz de convolução transposta
        //         }
        //         post(""); //nova linha após cada linha da matriz
        //     }
        // }
        //![DEPURAÇÃO]
    }
}

//*----------------------------------- ROTACIONA KERNEL 180º --------------------------------
static void create_rotated_kernel(t_cnn2d *x, t_float **original, t_float **rotated, int rows, int cols) {
    for(int i = 0; i < rows; i++) { //percorre as linhas do kernel
        for(int j = 0; j < cols; j++) { //percorre as colunas do kernel
            rotated[i][j] = original[rows-1-i][cols-1-j]; //rotaciona o kernel 180 graus e armazena no kernel rotacionado
            // post("Rotated kernel: %0.4f", rotated[i][j]); //imprime o kernel rotacionado
        }
        // post(""); //nova linha após cada linha do kernel
    }
}
 
//*-----------------------------------------------  BACKPROPAGATION DO DECODER ------------------------------------------------
static void backpropagation_decoder(t_cnn2d *x) {
    // índice da última camada do decoder
    int last_layer = x->num_Tlayers - 1;

    float perda_exemplo = 0.0; //variável para armazenar o erro acumulado de cada exemplo
    
    //* 1. CALCULO DO DELTA INICIAL E ERRO DA SAÍDA DA REDE 
    int out_rows = x->Tconv_size[last_layer*2]; //nº de linhas da matriz de saída (= nº de linhas da matriz de dados)
    int out_cols = x->Tconv_size[last_layer*2+1]; //nº de colunas da matriz de saída (= nº de colunas da matriz de dados)

    int n = out_rows * out_cols; //nº de elementos da matriz de saída

    //* - Obter a derivada da função de ativação para a camada de saída
    activation_derivative_func activation_deriv_last = get_activation_derivative_function(x->activation_function_c[last_layer + x->num_layers]);
    // post("Decoder Layer %d: Activation function: %s", last_layer + 1, x->activation_function_c[last_layer + x->num_layers]->s_name); //imprime a função de ativação [DEPURAÇÃO]
    // post(""); //linha em branco [DEPURAÇÃO]

    //ATENÇÃO: a função de erro é usada para calcular o erro da saída da rede e monitorar o treinamento
    //* - Obter a função de perda escolhida (MSE, MAE ou BCE)
    perda_func error_out = get_perda_function(x->error_function);
   
    //ATENÇÃO: a derivada da função de erro é usada apenas para calcular o delta da saída da rede
    //* - Obter a derivada da função de perda escolhida (MSE, MAE ou BCE) 
    perda_derivada deriv_error = get_perda_deriv(x->error_function);

    //* 1.1. Percorre a matriz de saída da rede
    int padding = x->padding[0]; //padding da primeira camada do encoder
    for(int i = 0; i < out_rows; i++) { 
        for(int j = 0; j < out_cols; j++) {
            int in_i = i + padding; //posição inicial da linha na matriz de entrada
            int in_j = j + padding; //posição inicial da coluna na matriz de entrada

            //* 1.2. calcula a perda da saída da rede a partir da função de erro escolhida apenas para monitoramento (MSE, MAE ou BCE)
            float perda = error_out(x, x->Tconv[last_layer][i][j], x->input[0][in_i][in_j]);
            perda_exemplo += perda; //acumula o erro de cada elemento da matriz de saída

            //* 1.3. calcula o delta da saída da rede
            //para ativação linear, o delta da saída da rede = derivada da função de perda utilizada
            if(x->activation_function_c[last_layer + x->num_layers] == gensym("linear")) {
                x->Tdelta[last_layer][i][j] = deriv_error(x, x->Tconv[last_layer][i][j], x->input[0][in_i][in_j]); //cálculo do delta da saída da rede para ativação linear
            }
            //para ativação sigmoid e função de perda BCE, o delta da saída da rede = derivada da função de perda utilizada
            else if(x->activation_function_c[last_layer + x->num_layers] == gensym("sigmoid") && x->error_function == gensym("bce")) {
                x->Tdelta[last_layer][i][j] = deriv_error(x, x->Tconv[last_layer][i][j], x->input[0][in_i][in_j]); //cálculo do delta da saída da rede para ativação sigmoid e função de perda BCE
            }
            //para demais ativações e funções de perda, o delta da saída da rede = derivada do erro * derivada da função de ativação
            else {
                x->Tdelta[last_layer][i][j] = deriv_error(x, x->Tconv[last_layer][i][j], x->input[0][in_i][in_j]) * activation_deriv_last(x->Tconv[last_layer][i][j], 0); //cálculo do delta da saída da rede
            }
        }
        //![DEPURAÇÃO]
        // //[DEPURAÇÃO] imprime o delta da saída da rede
        // post("[DEPURACAO 1.3.]Decoder Layer %d: Delta output", last_layer + 1);
        // for(int j = 0; j < out_cols; j++) { //percorre as colunas da matriz de saída
        //     post("%f ", x->Tdelta[last_layer][i][j]); //imprime o valor do delta da saída da rede
        // }
        //![DEPURAÇÃO]
    }
    //* 1.4. acumulação do erro para cada exemplo
    //divide o erro acumulado de cada elemento do exemplo atual pelo nº de elementos da matriz de saída
    x->erro_total = perda_exemplo / n; //acumula o valor dividido pelo nº de elementos da matriz de saída
    
    //* 2. PROPAGAÇÃO DO ERRO PARA AS CAMADAS DO DECODER DE TRÁS PARA FRENTE 
    for(int k = x->num_Tlayers-1; k >= 0; k--) { //percorre as camadas do decoder de trás para frente
        //recupera as dimensões do kernel, matriz de entrada, stride, padding da camada atual
        int kernel_rows = x->Tkernels_size[k*2]; //nº de linhas do kernel da camada atual
        int kernel_cols = x->Tkernels_size[k*2+1]; //nº de colunas do kernel da camada atual
        int input_rows = x->Tinput_padded[k*2]; //nº de linhas da matriz de entrada da camada atual
        int input_cols = x->Tinput_padded[k*2+1]; //nº de colunas da matriz de entrada da camada atual
        int stride_rows = x->Tstride_conv[k*2]; //stride de convolução para as linhas
        int stride_cols = x->Tstride_conv[k*2+1]; //stride de convolução para as colunas
        // int padding = x->Tpadding[k]; //padding da camada atual

        //* 2.1 CÁLCULO DO GRADIENTE DO KERNEL DA CAMADA ATUAL
        for(int kr = 0; kr < kernel_rows; kr++) { //percorre as linhas do kernel
            for(int kc = 0; kc < kernel_cols; kc++) { //percorre as colunas do kernel
                float grad = 0.0; //inicializa o gradiente com zero

                //percorre a matriz de entrada da camada atual (mapa de ativações da camada anterior)
                for(int i = 0; i < input_rows; i++) { //percorre as linhas da matriz de entrada da camada atual (mapa de ativações)
                    for(int j = 0; j < input_cols; j++) { //percorre as colunas da matriz de entrada da camada atual (mapa de ativações)
                        int out_i = i * stride_rows + kr - padding; //posição inicial da linha na matriz de saída
                        int out_j = j * stride_cols + kc - padding; //posição inicial da coluna na matriz de saída
                        if(out_i >= 0 && out_i < x->Tconv_size[k*2] && out_j >= 0 && out_j < x->Tconv_size[k*2+1]) { //verifica se a posição da matriz de saída é válida
                            grad += x->Tinput[k][i][j] * x->Tdelta[k][out_i][out_j]; //correlação cruzada entre a matriz de entrada da camada atual e o delta da saída da camada atual
                        }
                    }
                }
                //!ATENÇÃO: cada matriz Tkernel_grad tem o mesmo tamanho do kernel (ALOCAR MEMÓRIA)
                x->Tkernel_grad[k][kr][kc] = grad; //armazena o gradiente do kernel da camada atual em x->Tkernel_grad

            }
        }
        //! [DEPURAÇÃO]
        // //imprime o gradiente do kernel da camada atual
        // post("[DEPURACAO 2.1.]Decoder Layer %d: Kernel gradient [mesmo tamanho do kernel]", k + 1);
        // for(int i = 0; i < kernel_rows; i++) { //percorre as linhas do kernel
        //     for(int j = 0; j < kernel_cols; j++) { //percorre as colunas do kernel
        //         post("%f ", x->Tkernel_grad[k][i][j]); //imprime o valor do gradiente do kernel
        //     }
        //     post(""); //nova linha após cada linha do kernel
        // }
        //![DEPURAÇÃO]

        //* 2.2 CÁLCULO DO DELTA PARA A CAMADA ANTERIOR, SE A CAMADA ATUAL NÃO FOR A PRIMEIRA DO DECODER
        // delta da camada anterior é obtido pela convolução transposta do delta da camada atual com o kernel da camada atual rotacionado 180°
        if(k > 0) { //verifica se a camada atual não é a primeira camada do decoder
            // Rotacionar o kernel 180°
            create_rotated_kernel(x, x->Tkernel[k], x->Tkernel_rotated[k], kernel_rows, kernel_cols);

            //dimensões da matriz de saída da camada anterior
            int prev_conv_rows = x->Tconv_size[(k-1)*2]; //nº de linhas da matriz de saída da camada anterior
            int prev_conv_cols = x->Tconv_size[(k-1)*2+1]; //nº de colunas da matriz de saída da camada anterior
            
            // Inicializa delta da camada anterior
            for(int i = 0; i < prev_conv_rows; i++) {
                for(int j = 0; j < prev_conv_cols; j++) {
                    x->Tdelta[k-1][i][j] = 0.0; //inicializa o delta da camada anterior com zero
                }
            }
            // Convolução transposta com kernel rotacionado
            for(int i = 0; i < x->Tconv_size[k*2]; i++) { //percorre as linhas da matriz de saída da camada atual
                for(int j = 0; j < x->Tconv_size[k*2+1]; j++) { //percorre as colunas da matriz de saída da camada atual
                    for(int kr = 0; kr < kernel_rows; kr++) { //percorre as linhas do kernel da camada atual
                        for(int kc = 0; kc < kernel_cols; kc++) { //percorre as colunas do kernel da camada atual
                            int prev_i = (i - kr + padding) / stride_rows; //posição inicial da linha na matriz de saída da camada anterior
                            int prev_j = (j - kc + padding) / stride_cols; //posição inicial da coluna na matriz de saída da camada anterior

                            // Verifica se a posição da matriz de saída da camada anterior é válida
                            if(prev_i >= 0 && prev_j >= 0 && 
                               prev_i < prev_conv_rows && prev_j < prev_conv_cols &&
                               (i - kr + padding) % stride_rows == 0 &&
                               (j - kc + padding) % stride_cols == 0) {
                                x->Tdelta[k-1][prev_i][prev_j] += x->Tkernel_rotated[k][kr][kc] * x->Tdelta[k][i][j]; //convolução transposta do delta da saída da camada atual com o kernel rotacionado 180°
                            } //TODO: TESTAR A ROTAÇÃO DO KERNEL SEM ARMZENAR EM UMA NOVA MATRIZ
                        }
                    }
                }
            }
            // Aplica derivada da ativação
            activation_derivative_func deriv_prev = get_activation_derivative_function(x->activation_function_c[(k-1)+x->num_layers]); //função de ativação da camada anterior (k-1+num_layers)

            //! [DEPURAÇÃO]
            // //imprime a função de ativação da camada anterior
            // post("[DEPURACAO 2.2.]Decoder Layer %d: Activation function = %s", k, x->activation_function_c[(k-1)+x->num_layers]->s_name);
            // post("");
            //! [DEPURAÇÃO]
            
            // multiplica o delta da camada anterior pela derivada da função de ativação da camada anterior
            for(int i = 0; i < prev_conv_rows; i++) {
                for(int j = 0; j < prev_conv_cols; j++) {
                    //ATENÇÃO: cada matriz de delta tem o mesmo tamanho da matriz de saída de cada camada (ALOCAR MEMÓRIA)
                    x->Tdelta[k-1][i][j] *= deriv_prev(x->Tconv[k-1][i][j], 0); //multiplica pela derivada da função de ativação da camada anterior
                }
            }

            //! [DEPURAÇÃO]
            // //imprime o delta da camada anterior
            // post("[DEPURACAO 2.2.]Decoder Layer %d: Delta [mesmo tamanho da conv trans]", k);
            // for(int i = 0; i < prev_conv_rows; i++) { //percorre as linhas da matriz de saída da camada anterior
            //     for(int j = 0; j < prev_conv_cols; j++) { //percorre as colunas da matriz de saída da camada anterior
            //         post("%f ", x->Tdelta[k-1][i][j]); //imprime o valor do delta da camada anterior
            //     }
            //     post(""); //nova linha após cada linha da matriz
            // }
            //! [DEPURAÇÃO]
        }

        //* 2.3 SE A CAMADA ATUAL É A PRIMEIRA DO DECODER, CALCULA O DELTA DA ÚLTIMA CAMADA DO ENCODER
        //? OBS: A MATRIZ DE DELTA DA ÚLTIMA CAMADA DO ENCODER DEVE TER AS MESMAS DIMENSÕES DA MATRIZ DE POOLING DA ÚLTIMA CAMADA DO ENCODER
        else { // Caso especial para a primeira camada do decoder (k = 0)
            // Rotacionar kernel 180°
            create_rotated_kernel(x, x->Tkernel[k], x->Tkernel_rotated[k], kernel_rows, kernel_cols);
            // t_float **rotated_kernel = create_rotated_kernel(x, x->Tkernel[k], kernel_rows, kernel_cols);
            //dimensões da matriz de pooling da última camada do encoder
            int prev_pool_rows = x->pooling_matriz_size[(x->num_layers-1)*2]; //nº de linhas da matriz de pooling da última camada do encoder
            int prev_pool_cols = x->pooling_matriz_size[(x->num_layers-1)*2+1]; //nº de colunas da matriz de pooling da última camada do encoder
            
            // Inicializa delta da última camada do encoder
            for(int i = 0; i < prev_pool_rows; i++) {
                for(int j = 0; j < prev_pool_cols; j++) {
                    x->delta[x->num_layers-1][i][j] = 0.0; //inicializa o delta da última camada do encoder com zero
                }
            }
            // Convolução transposta com kernel rotacionado
            for(int i = 0; i < x->Tconv_size[k*2]; i++) { //percorre as linhas da matriz de saída da primeira camada do decoder
                for(int j = 0; j < x->Tconv_size[k*2+1]; j++) { //percorre as colunas da matriz de saída da primeira camada do decoder
                    for(int kr = 0; kr < kernel_rows; kr++) { //percorre as linhas do kernel da primeira camada do decoder
                        for(int kc = 0; kc < kernel_cols; kc++) { //percorre as colunas do kernel da primeira camada do decoder
                            //ATENÇÃO: VERIFICAR ESSAS DIMENSÕES
                            int prev_i = (i - kr + padding) / stride_rows; //posição inicial da linha na matriz de pooling da última camada do encoder
                            int prev_j = (j - kc + padding) / stride_cols; //posição inicial da coluna na matriz de pooling da última camada do encoder

                            // Verifica se a posição da matriz pooling da última camada do encoder é válida
                            if(prev_i >= 0 && prev_j >= 0 && prev_i < prev_pool_rows && prev_j < prev_pool_cols && (i - kr + padding) % stride_rows == 0 && (j - kc + padding) % stride_cols == 0) {
                                x->delta[x->num_layers-1][prev_i][prev_j] += x->kernel_rotated[k][kr][kc] * x->Tdelta[k][i][j]; //convolução transposta do delta da saída da camada atual com o kernel rotacionado 180°
                            }
                        }
                    }
                }
            }
            //! ATENÇÃO: APLICAÇÃO DA DERIVADA DA FUNÇÃO DE ATIVAÇÃO DA ÚLTIMA CAMADA DO ENCODER É FEITA APÓS A PROPAGAÇÃO DO GRADIENTE DO POOLING

            //! [DEPURAÇÃO]
            // //imprime o delta da última camada do encoder
            // post("[DEPURACAO 2.3.]Encoder Layer %d: Delta [mesmo tamanho da saída do encoder]", x->num_layers);
            // for(int i = 0; i < x->pooling_matriz_size[(x->num_layers-1)*2]; i++) { //percorre as linhas da matriz de pooling da última camada do encoder
            //     for(int j = 0; j < x->pooling_matriz_size[(x->num_layers-1)*2+1]; j++) { //percorre as colunas da matriz de pooling da última camada do encoder
            //         post("%f ", x->delta[x->num_layers-1][i][j]); //imprime o valor do delta da última camada do encoder
            //     }
            //     post(""); //nova linha após cada linha da matriz
            // }
            //![DEPURAÇÃO]
        }
    }
}

//*---------------------------------- PROPAGA O DELTA PELO DO MAX POOLING  ----------------------------------------------
static void max_pooling_grad(t_float** conv_matrix, t_float** pool_matrix, // Matrizes de convolução (saída) e pooling (entrada)
                     t_float** delta_pool, t_float** delta_conv, // Deltas de pooling (entrada) e convolução (saída)
                     int pool_rows, int pool_cols, // Dimensões de delta_pool
                     int pool_size_rows, int pool_size_cols, // Dimensões da janela de pooling
                     int stride_rows, int stride_cols) { // Stride de pooling
    
    // Inicializa o delta da matriz de convolução com zeros
    int conv_rows = pool_rows * stride_rows;
    int conv_cols = pool_cols * stride_cols;
    for(int i = 0; i < conv_rows; i++) {
        for(int j = 0; j < conv_cols; j++) {
            delta_conv[i][j] = 0.0;
        }
    }
    // Propaga os gradientes apenas para as posições máximas
    for(int i = 0; i < pool_rows; i++) {
        for(int j = 0; j < pool_cols; j++) {
            // Encontra posição do máximo
            int max_i = i * stride_rows;
            int max_j = j * stride_cols;
            float max_val = conv_matrix[max_i][max_j];
            
            for(int m = 0; m < pool_size_rows; m++) {
                for(int n = 0; n < pool_size_cols; n++) {
                    int curr_i = i * stride_rows + m;
                    int curr_j = j * stride_cols + n;
                    if(conv_matrix[curr_i][curr_j] > max_val) {
                        max_val = conv_matrix[curr_i][curr_j];
                        max_i = curr_i;
                        max_j = curr_j;
                    }
                }
            }
            delta_conv[max_i][max_j] = delta_pool[i][j];
        }
    }
}

//*---------------------------------- FUNÇÃO PARA PROPAGAR O GRADIENTE DO AVERAGE POOLING  ----------------------------------------------
//arggs: gradiente da camada de pooling (entrada), gradiente da camada de convolução (saída), dimensões de delta_pool, dimensões da janela de pooling, stride de pooling, dimensões de delta_conv
static void average_pooling_grad(t_float** delta_pool, // Gradiente da camada de pooling (entrada)
                          t_float** delta_conv, // Gradiente da camada de convolução (saída)
                          int pool_rows, int pool_cols, // Dimensões de delta_pool
                          int pool_size_rows, int pool_size_cols, // Dimensões da janela de pooling
                          int stride_rows, int stride_cols, // Stride de pooling
                          int conv_rows, int conv_cols) {  // Dimensões de delta_conv

    // Inicializa delta_conv com zeros
    for(int i = 0; i < conv_rows; i++) {
        for(int j = 0; j < conv_cols; j++) {
            delta_conv[i][j] = 0.0;
        }
    }

    // Distribui o gradiente de delta_pool para delta_conv
    for(int i = 0; i < pool_rows; i++) {
        for(int j = 0; j < pool_cols; j++) {
            float grad = delta_pool[i][j] / (pool_size_rows * pool_size_cols);
            
            for(int m = 0; m < pool_size_rows; m++) {
                for(int n = 0; n < pool_size_cols; n++) {
                    int conv_i = i * stride_rows + m;
                    int conv_j = j * stride_cols + n;
                    
                    // Verifica limites para evitar acesso inválido
                    if (conv_i < conv_rows && conv_j < conv_cols) {
                        delta_conv[conv_i][conv_j] += grad;  // Acumula o gradiente
                    }
                }
            }
        }
    }
}


//*------------------------------------------- BACKPORPAGATION DO ENCODER --------------------------------------------------------
static void backpropagation_encoder(t_cnn2d *x) {
     //! ATEÇÃO: A MATRIZ DE DELTA DA ÚLTIMA CAMADA DO ENCODER ESTÁ SENDO CALCULADA DENTRO DA FUNÇÃO DE BACKPROPAGATION DO DECODER

    //![DEPURAÇÃO] 
    // //imprime o delta da última camada do encoder
    // post("[DEPURACAO]Encoder Layer %d: Delta [mesmo tamanho da matriz de pooling]", x->num_layers);
    // for(int i = 0; i < x->pooling_matriz_size[(x->num_layers-1)*2]; i++) { //percorre as linhas da matriz de pooling da última camada do encoder
    //     for(int j = 0; j < x->pooling_matriz_size[(x->num_layers-1)*2+1]; j++) { //percorre as colunas da matriz de pooling da última camada do encoder
    //         post("%f ", x->delta[x->num_layers-1][i][j]); //imprime o valor do delta da última camada do encoder
    //     }
    //     post(""); //nova linha após cada linha da matriz
    // }
    //![DEPURAÇÃO]

    //* PROPPAGA O DELTA PARA AS CAMADAS DO ENCODER DE TRÁS PARA FRENTE
    for(int k = x->num_layers-1; k >= 0; k--) { //percorre as camadas do encoder de trás para frente

        int kernel_rows = x->kernels_size[k*2];//nº de linhas do kernel da camada atual
        int kernel_cols = x->kernels_size[k*2+1]; //nº de colunas do kernel da camada atual

        int input_rows = x->input_padded[k*2]; //nº de linhas da matriz de entrada da camada atual
        int input_cols = x->input_padded[k*2+1]; //nº de colunas da matriz de entrada da camada atual

        int stride_rows = x->stride_conv[k*2]; //stride de convolução para as linhas
        int stride_cols = x->stride_conv[k*2+1]; //stride de convolução para as colunas

        int pool_stride_rows = x->stride_pool[k*2]; //stride de pooling das linhas da camada atual
        int pool_stride_cols = x->stride_pool[k*2+1]; //stride de pooling das colunas da camada atual

        int padding = x->padding[k]; //padding da camada atual

        //* 1. PROPAGA O DELTA DA CAMADA ANTERIOR ATRAVÉS DO POOLING (MAX OU AVG)
        int pool_rows = x->pooling_size[k*2]; //nº de linhas da janela de pooling da camada atual
        int pool_cols = x->pooling_size[k*2+1]; //nº de colunas da janela de pooling da camada atual

        // recupera o método de pooling configurado (max ou avg)
        t_symbol *pooling_method = x->pooling_function[k]; //recupera o método de pooling da camada atual

        //se o método de pooling for max
        if(pooling_method == gensym("max")) { 
            //atribui o gradiente apenas para as posições máximas
            max_pooling_grad(x->convolution[k], x->pooling[k], //matriz de convolução, matriz de pooling 
                               x->delta[k], x->conv_grad[k], //delta da matriz de pooling (entrada), delta da matriz de convolução (saída)
                               x->pooling_matriz_size[k*2], // linhas da matriz de pooling
                               x->pooling_matriz_size[k*2+1], // colunas da matriz de pooling
                               pool_rows, pool_cols, // linhas e colunas da janela de pooling
                               x->stride_pool[k*2], // stride de pooling para as linhas
                               x->stride_pool[k*2+1]); // stride de pooling para as colunas
        } 
        //se o método de pooling for avg
        else if(pooling_method == gensym("avg")) {
            //arggs: gradiente da camada de pooling (entrada), gradiente da camada de convolução (saída), dimensões de delta_pool, dimensões da janela de pooling, stride de pooling, dimensões de delta_conv
            //distribui o gradiente igualmente para todas as posições da janela
            average_pooling_grad(x->delta[k], x->conv_grad[k],//delta da matriz de convolução
                               x->pooling_matriz_size[k*2], //linhas da matriz de pooling
                               x->pooling_matriz_size[k*2+1], //colunas da matriz de pooling
                               pool_rows, pool_cols, //linhas e colunas da janela de pooling
                               x->stride_pool[k*2], //stride de pooling para as linhas
                               x->stride_pool[k*2+1],
                               x->convolv_matriz_size[k*2], x->convolv_matriz_size[k*2+1]); //stride de pooling para as colunas
        } 

        //* 2. MULTIPLICA O DELTA OBTIDO NA PROPAGAÇÃO ATRAVÉS DO POOLING PELA DERIVADA DA FUNÇÃO DE ATIVAÇÃO
        // Recupera a função de ativação da camada atual
        activation_derivative_func deriv = get_activation_derivative_function(x->activation_function_c[k]);
        
        //![DEPURAÇÃO]
        // // percorre a matriz de convolução da camada atual
        // post("[DEPURACAO 2.]Encoder Layer %d: Convolution gradient[mesmo tamanho da conv]", k + 1); //imprime o gradiente da matriz de convolução da camada atual
        //![DEPURAÇÃO]

        for(int i = 0; i < x->convolv_matriz_size[k*2]; i++) { //percorre as linhas da matriz de convolução da camada atual
            for(int j = 0; j < x->convolv_matriz_size[k*2+1]; j++) { //percorre as colunas da matriz de convolução da camada atual
                x->conv_grad[k][i][j] *= deriv(x->convolution[k][i][j], 0); //multiplica o gradiente propagado pelo pooling pela derivada da função de ativação e armazena em x->conv_grad 
                
                //![DEPURAÇÃO]
                // //[DEPURAÇÃO] imprime o gradiente da matriz de convolução de cada camada do encoder
                // post("%0.4f", x->conv_grad[k][i][j]); //imprime o gradiente da matriz de convolução de cada camada do encoder
                //![DEPURAÇÃO]
            }
        }

        //* 3. CALCULA OS GRADIENTES DO KERNEL
        // Inicializa os gradientes com zeros
        for(int kr = 0; kr < kernel_rows; kr++) {
            for(int kc = 0; kc < kernel_cols; kc++) {
                float grad = 0.0;
                // Percorre a matriz de entrada da camada atual
                for(int i = 0; i < input_rows; i++) { //percorre as linhas da matriz de entrada da camada atual
                    for(int j = 0; j < input_cols; j++) { //percorre as colunas da matriz de entrada da camada atual
                        int out_i = (i - kr + padding) / stride_rows; //posição inicial da linha na matriz de convolução da camada atual
                        int out_j = (j - kc + padding) / stride_cols; //posição inicial da coluna na matriz de convolução da camada atual
                        
                        // Verifica se a posição da matriz de convolução é válida
                        if(out_i >= 0 && out_j >= 0 &&
                           out_i < x->convolv_matriz_size[k*2] && 
                           out_j < x->convolv_matriz_size[k*2+1]) {
                            grad += x->input[k][i][j] * x->conv_grad[k][out_i][out_j]; //correlação cruzada entre a matriz de entrada da camada atual e o gradiente da matriz de convolução
                        }
                    }
                }
                x->kernels_grad[k][kr][kc] = grad; //armazena o gradiente do kernel da camada atual em x->kernels_grad
            }
        }

        //![DEPURAÇÃO]
        // //imprime o gradiente do kernel de cada camada do encoder
        // post("[DEPURACAO 3.]Encoder Layer %d: Kernel gradient [mesmo tamanho do kernel]", k + 1); //imprime o nº da camada atual
        // for(int kr = 0; kr < kernel_rows; kr++) { //percorre as linhas do kernel da camada atual
        //     for(int kc = 0; kc < kernel_cols; kc++) { //percorre as colunas do kernel da camada atual
        //         post("%0.4f", x->kernels_grad[k][kr][kc]); //imprime o gradiente do kernel de cada camada do encoder
        //     }
        // }
        //![DEPURAÇÃO]

        //* 4. CALCULO DO DELTA PARA A CAMADA ANTERIOR K-1 
        //ATENÇÃO!!! DELTA DA CAMADA ANTERIOR K-1 TEM O MESMO TAMANHO DA MATRIZ DE POOLING DA CAMADA ANTERIOR K-1
        if(k > 0) { //verifica se a camada atual não é a primeira camada do encoder. Se for, não há delta para calcular (fim do backpropagation do encoder)

            int prev_layer = k - 1; //índice da camada anterior
            
            int prev_rows = x->pooling_matriz_size[prev_layer*2]; //nº de linhas da matriz de pooling da camada anterior
            int prev_cols = x->pooling_matriz_size[prev_layer*2+1]; //nº de colunas da matriz de pooling da camada anterior
            
            // int prev_stride_rows = x->stride_conv[prev_layer*2]; //stride de convolução para as linhas da camada anterior
            // int prev_stride_cols = x->stride_conv[prev_layer*2+1]; //stride de convolução para as colunas da camada anterior

            int prev_padding = x->padding[prev_layer]; //padding da camada anterior
            
            // Inicializa o delta da camada anterior com zeros
            for(int i = 0; i < prev_rows; i++) { //percorre as linhas da matriz de pooling da camada anterior
                for(int j = 0; j < prev_cols; j++) { //percorre as colunas da matriz de pooling da camada anterior
                    x->delta[prev_layer][i][j] = 0.0; //inicializa o delta da camada anterior com zero
                }
            }

            // Rotaciona o kernel da camada atual 180°
            create_rotated_kernel(x, x->kernels[k], x->kernel_rotated[k], kernel_rows, kernel_cols);
            
            //*  CONVOLUÇÃO DO DELTA DA CONVOLUÇÃO COM O KERNEL ROTACIONADO 180° 
            // Percorre a matriz de convolução da camada atual
            for(int i = 0; i < x->convolv_matriz_size[k*2]; i++) {
                for(int j = 0; j < x->convolv_matriz_size[k*2+1]; j++) {
                    // Percorre o kernel da camada atual rotacionado
                    for(int kr = 0; kr < kernel_rows; kr++) {
                        for(int kc = 0; kc < kernel_cols; kc++) {
                            int in_i = i * stride_rows + kr - padding; //posição inicial da linha na matriz de entrada da camada anterior
                            int in_j = j * stride_cols + kc - padding; //posição inicial da coluna na matriz de entrada da camada anterior
                            
                            if(in_i >= 0 && in_i < prev_rows && in_j >= 0 && in_j < prev_cols) {
                                x->delta[k-1][in_i][in_j] += x->kernel_rotated[k][kr][kc] * x->conv_grad[k][i][j]; //convolução do delta da camada atual com o kernel rotacionado 180° da camada atual = delta da camada anterior
                            } //TODO: TESTAR A ROTAÇÃO DO KERNEL SEM ARMZENAR EM UMA NOVA MATRIZ
                        }
                    }
                }
            }
            //![DEPURAÇÃO] 
            // //imprime o delta da camada anterior
            // post("[DEPURACAO 4.]Encoder Layer %d: Delta", k); //imprime o nº da camada atual
            // for(int i = 0; i < x->pooling_matriz_size[(k-1)*2]; i++) { //percorre as linhas da matriz de entrada da camada atual
            //     for(int j = 0; j < x->pooling_matriz_size[(k-1)*2+1]; j++) { //percorre as colunas da matriz de entrada da camada atual
            //         post("%lf ", x->delta[k-1][i][j]); //imprime o valor do delta da camada atual
            //     }
            //     post(""); //nova linha após cada linha da matriz
            // }
            //![DEPURAÇÃO]
        }
    }
}

//*-------------------------------------- MÉTRICA: ERRO MÉDIO ABSOLUTO RELATIVO POR FEATURE ----------------------------------------------
static float relative_error_per_descriptor(t_cnn2d *x, int feature_index) { //argumento: nº de featurea (nº de linhas da matriz de entrada)
    //*1. recupera o nº de elementos de cada feature (nº de colunas da matriz original) e inicializa as variáveis
    int analise = (int)x->input_size[1]; // Número de colunas (elementos de cada feature) na matriz original
    int count = 0;
    float error_sum = 0.0;

    //* 2. Para cada linha (feature), percorre cada coluna (elemento de cada feature) (linha = descriptor_index, coluna = j)
    for (int j = 0; j < analise; j++) {
        float original = x->input[0][feature_index][j];// para cada feature, recupera o elemento j da matriz de dados 
        float reconstructed = x->Tconv[x->num_Tlayers - 1][feature_index][j]; // para cada feature, recupera o elemento da matriz de dados reconstruída 
        
        //* 3. Calcula o erro relativo médio para cada feature (erro absoluto médio relativo)
        if (fabs(original) > 1e-6) { // Evita divisão por zero
            error_sum += fabs((original - reconstructed) / original); //soma o erro absoluto médio relativo para cada feature 
            count++; //incrementa o contador de features válidas
        }
    }
    
    //* 4. Retorna o erro médio relativo por feature (se não houver amostras válidas, retorna 0)
    return (count > 0) ? (100 * error_sum / count) : 0.0;
}


//*-------------------------------------- MÉTRICA: CORRELAÇÃO DE PEARSON ---------------------------------------------------
 // Calcula a correlação de Pearson para um descritor específico. ver: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
static float pearson_correlation(t_cnn2d *x, int feature_index) { //argumento: nº de elementos de cada feature (nº da coluna da matriz de entrada)

    //* 1. recupera o nº de elementos de cada feature (nº de colunas da matriz de entrada) e inicializa as variáveis
    int analise = (int)x->input_size[1]; // Número de colunas (elementos de cada feature) da matriz de entrada;
    float sum_orig = 0.0, sum_recon = 0.0; //inicializa a soma dos valores originais e reconstruídos com zero
    float sum_orig_recon = 0.0, sum_orig_sq = 0.0, sum_recon_sq = 0.0; //inicializa a soma dos valores originais e reconstruídos ao quadrado com zero

    //* 2. Para cada linha (feature), percorre cada coluna (elemento de cada feature) (linha = feature_index, coluna = j)
    for(int j = 0; j < analise; j++) { //para cada feature (linha) percorre cada elemento (coluna)
        float orig = x->input[0][feature_index][j]; //para cada feature, recupera o elemento j da matriz de dados
        float recon = x->Tconv[x->num_Tlayers-1][feature_index][j]; //para cada feature, recupera o elemento j da matriz de dados reconstruída
        
        //* 3. soma os valores originais, reconstruídos, o produto dos valores originais e reconstruídos, o quadrado dos valores originais e o quadrado dos valores reconstruídos
        sum_orig += orig; //soma os valores originais
        sum_recon += recon; //soma os valores reconstruídos
        sum_orig_recon += orig * recon; //soma o produto dos valores originais e reconstruídos
        sum_orig_sq += orig * orig; //soma o quadrado dos valores originais
        sum_recon_sq += recon * recon; //soma o quadrado dos valores reconstruídos
    }

    //* 4. Calcula a correlação de Pearson
    float numerator = sum_orig_recon - (sum_orig * sum_recon) / analise; 
    float denom_orig = sum_orig_sq - (sum_orig * sum_orig) / analise;
    float denom_recon = sum_recon_sq - (sum_recon * sum_recon) / analise;
    
    if(denom_orig <= 0 || denom_recon <= 0) return 0.0; // Evita divisão por zero
    
    return numerator / sqrt(denom_orig * denom_recon); //retorna a correlação de Pearson por exemplo de teste 
}


//*-------------------------------------------- AVALIAÇÃO DO MODELO ----------------------------------------------------- 
static void evaluation (t_cnn2d *x, int n) {
    float perda_exemplo = 0.0; //inicializa a perda do exemplo com zero
    float perda = 0; //inicializa a perda com zero
    //* 1. ENCODER: convolução -> ativação -> pooling
    convolution_pooling(x);

    //* 2. DECODER: convolução transposta -> ativação -> corte do padding
    transposed_convolution(x);

    //* 3. CÁLCULO DO ERRO DO MODELO 
    //* 3.1. Obter a função de perda escolhida (MSE, MAE ou BCE)
    perda_func error_out = get_perda_function(x->error_function);

    //* 3.2. calcula o erro para um exemplode teste a partir da função de perda escolhida (MSE, MAE ou BCE)
    for (int i = 0; i < x->Tconv_size[(x->num_Tlayers - 1)*2]; i++) { //percorre as linhas da matriz de saída da rede
        int in_i = i + x->padding[0]; //posição inicial da linha na matriz de entrada
        for (int j = 0; j < x->Tconv_size[(x->num_Tlayers - 1)*2+1]; j++) { //percorre as colunas da matriz de saída da rede
            int in_j = j + x->padding[0]; //posição inicial da coluna na matriz de entrada
            perda = error_out(x, x->Tconv[x->num_Tlayers - 1][i][j], x->input[0][in_i][in_j]);
            perda_exemplo += perda; //acumula o erro de cada elemento da matriz de saída
        }
    }
    //*3.3. Calcula a média do erro para cada exemplo e acumula
    post("Example %d - mean error %0.4f", x->current_data, (perda_exemplo/n)); //imprime o erro total
    x->erro_total += perda_exemplo / n; //acumula a média do erro para cada exemplo
    // for (int d = 0; d < num_descriptors; d++) {
    // //* 3.2. Calcula o erro relativo médio por feature
    //     float rel_error = relative_error_per_descriptor(x, d); //calcula o erro relativo médio por feature
    //     float rel_accuracy = 100.0 - rel_error; //calcula o acerto relativo médio por feature
    //     //* 3.3. Calcula a correlação de Pearson por feature
    //     float pearson = pearson_correlation(x, d); //calcula a correlação de Pearson por feature
    //     //* 3.4. imprime as métricas de avaliação por feature
    //     post("Feature %d: Relative accuracy: %lf%% - Relative Error: %lf%%", d + 1, rel_accuracy, rel_error);
    //     post("\n"); //nova linha
    //     post("Feature %d: Pearson correlation %lf", d + 1, pearson);
    // }
}

//* ------------------------------------ TREINAMENTO, AVALIAÇÃO, RECONSTRUÇÃO E EXTRAÇÃO DE FEATURES --------------------------------------------
static void input_data_prop(t_cnn2d *x, t_symbol *s, int argc, t_atom *argv) {
    //* 1. RECEBE UM VETOR COM OS DADOS DE ENTRADA
    if (argc != x->input_size[0] * x->input_size[1]) { //Verifica se o número de argumentos é compatível com o tamanho da matriz de entrada
        error("Please provide an input compatible with the data matrix: %d x %d", (int)x->input_size[0], (int)x->input_size[1]);
        return;
    }
    //*1.1. recuperar as dimensões da matriz de entrada com padding
    int linhas = (int)x->input_padded[0];  // recupera Nº de linhas da matriz com padding
    int colunas = (int)x->input_padded[1]; // Nº de colunas da matriz com padding
    int padding = (int)x->padding[0]; // padding da matriz de entrada da primeira camada
    
    // //* 2. TRANSFORMA O VETOR RECEBIDO EM UMA MATRIZ DE DADOS MANTENDO REGIÕES DE PADDING COM ZERO
    //?OBS: A MATRIZ DE DADOS JÁ FOI PREENCHIDA COM ZEROS NA FUNÇÃO MATRIX_ENCODER_SIZE PARA FACILITAR A MANUTENÇÃO DAS REGIÕES DE PADDING
    int k = 0; // Índice para percorrer o vetor argv
    for (int i = padding; i < linhas - padding; i++) { // percorre as Linhas ignorando a área de padding
        for (int j = padding; j < colunas - padding; j++) { // percorre a Colunas ignorando a área de padding
            x->input[0][i][j] = argv[k].a_w.w_float; // Atribui valores recebidos à matriz de dados
            k++; // Incrementa o índice do vetor argv
        }
    }

    //![DEPURAÇÃO]
    // convolution_pooling(x); // Propagação dos dados de entrada pelas camadas do encoder (convolução e pooling)

    // transposed_convolution(x); // Propagação dos dados de saída do encoder pelas camadas do decoder (convolução transposta)

    // backpropagation_decoder(x); // Backpropagation do decoder
    //!![DEPURAÇÃO]
    
    //* 3. TREINAMENTO: PROPAGAÇÃO DOS DADOS DE ENTRADA PELAS CAMADAS DO ENCODER (CONVOLUÇÃO E POOLING) E DECODER (CONVOLUÇÃO TRANSPOSTA)
    //somente se o modo de treinamento estiver ativado e o nº de épocas for menor que o nº máximo de épocas
    if(x->current_epoch < x->maxepochs && x->current_data < x->datasize && x->trainingmode_c == 1 && x->evalmode == 0){
        //*3.1. ENCODER: convolução -> ativação -> pooling
        convolution_pooling(x);
        //*3.2. DECODER: convolução transposta -> ativação -> corte do padding
        transposed_convolution(x);

        //* 4. BACKPROPAGATION DO DECODER E ENCODER (CÁLCULO DOS GRADIENTES)
        //*4.1. backpropagation do decoder
        backpropagation_decoder(x);
        //*4.2. backpropagation do encoder
        backpropagation_encoder(x);

        //* 5. ATUALIZAÇÃO DOS PESOS E BIAS COM O OTIMIZADOR ESCOLHIDO (SGD OU ADAM)
        //*5.1. verifica se o otimizador escolhido é o SGD
        if(x->optimizer == gensym("sgd")) {
            //*5.2. atualiza os pesos (kernels) e bias com o otimizador SGD
            sgd(x);
        }
        //*5.2. verifica se o otimizador escolhido é o ADAM
        else if(x->optimizer == gensym("adam")) {
            //*5.3. atualiza os pesos (kernels) e bias com o otimizador ADAM
            adam(x, x->current_epoch);
        }
        else {
            error("Please choose an optimizer (sgd or adam)");
        }
        x->current_data++; //incrementa o índice do exemplo de treinamento
        // post("Exemplo: %d", x->current_data); //imprime o índice do exemplo de treinamento
        
    }
    //* 6. CALCULA A MÉDIA DA PERDA ACUMULADA EM CADA ÉPOCA E INCREMENTA O CONTADOR DE ÉPOCAS (FIM DE UMA ÉPOCA)
    //se está no modo de treinamento e o índice do exemplo atual é igual ao número máximo de exemplos:
    //retorna o erro e o nº da época atual, adicona uma época ao contador e zera o índice do exemplo e o erro total
    if (x->current_epoch < x->maxepochs && x->current_data == x->datasize && x->trainingmode_c == 1) {
        float loss = 0.0; //erro médio da época
        //calcula o erro médio da época
        loss = x->erro_total / x->datasize; 
        //copia o nº da época atual para o buffer de saída
        SETFLOAT(x->epoch_out, x->current_epoch);
        //envia o nº da época atual para o outlet 3
        outlet_anything(x->x_out3, gensym("epoch"), 1, x->epoch_out);
        //copia o erro médio da época para o buffer de saída
        SETFLOAT(x->error_out, loss);
        //envia o erro médio da época para o outlet 3
        outlet_anything(x->x_out3, gensym("error"), 1, x->error_out);
        //imprime o erro médio da época
        post("Epoch %d - mean error: %lf", x->current_epoch, loss);

        x->current_epoch++; //incrementa o contador de épocas
        x->erro_total = 0.0; //zera o erro total da rede para a próxima época
        x->current_data = 0; //zera o índice do exemplo de treinamento para a próxima época
        // post("Epoch %d", x->current_epoch); //imprime o nº da época
    }

    //* 7. NÚMERO MÁXIMO DE ÉPOCAS ALCANÇADO (FIM DO TREINAMENTO)
    if(x->current_epoch == x->maxepochs && x->trainingmode_c == 1) {
        post("The training process has reached the maximum amount of epochs (%d/%d)", x->current_epoch, x->maxepochs);
    }

    //* 8. MODO DE RECONSTRUÇÃO DE DADOS E EXTRAÇÃO DE FEATURES (REDE TREINADA)
    //se não está no modo de treinamento e nem no modo de avaliação, apenas propaga os dados de entrada pelas camadas do encoder e decoder
    if(x->trainingmode_c == 0 && x->evalmode == 0) {
        // post("Reconstruction and feature extraction mode");
        //*8.1. ENCODER: convolução -> ativação -> pooling
        convolution_pooling(x);
        //*8.2. DECODER: convolução transposta -> ativação -> corte do padding
        transposed_convolution(x);

        //*8.3. RETORNA O ESPAÇO LATENTE NO OUTLET 1
        int latente_linhas = (int)x->pooling_matriz_size[(x->num_layers-1)*2]; //nº de linhas da matriz de entrada
        int latent_colunas = (int)x->pooling_matriz_size[(x->num_layers-1)*2+1]; //nº de colunas da matriz de entrada
        int latente_layer = x->num_layers - 1; //índice da última camada do encoder
        for (int i = 0; i < latente_linhas; i++) { //percorre as linhas da matriz de pooling da última camada do encoder
            for(int j = 0; j < latent_colunas; j++) { //percorre as colunas da matriz de pooling da última camada do encoder
            //preenche o vetor de saída com os valores da última camada do encoder
            SETFLOAT(x->latent_out + j, x->pooling[(latente_layer)][i][j]); //preenche o buffer de saída com os valores da última camada do encoder
            }
            // Envia o buffer de saída para o outlet 1
            outlet_list(x->x_out1, &s_list, latent_colunas, x->latent_out);
        }

        //*8.4. RETORNA A MATRIZ DE DADOS RECONSTRUÍDA NO OUTLET 2
        int output_linhas = (int)x->Tconv_size[(x->num_Tlayers-1)*2]; //nº de linhas da matriz de dados reconstruída
        int output_colunas = (int)x->Tconv_size[(x->num_Tlayers-1)*2+1]; //nº de colunas da matriz de dados reconstruída
        int output_layer = x->num_Tlayers - 1; //índice da última camada do decoder
        for (int i = 0; i < output_linhas; i++) { //percorre as linhas da matriz reconstruida
            for(int j = 0; j < output_colunas; j++) { //percorre as colunas da matriz reconstrúida
                //preenche o vetor de saída com os valores da matriz de dados reconstruída
                SETFLOAT(x->matrix_out + j, x->Tconv[output_layer][i][j]); //preenche o buffer de saída com os valores da matriz de dados reconstruída
            }
            // Envia o buffer de saída para o outlet 2
            outlet_list(x->x_out2, &s_list, x->Tconv_size[(x->num_Tlayers-1)*2+1], x->matrix_out);
        }
    }

    //* 9. MODO DE AVALIAÇÃO DO MODELO
    int n = x->input_size[1]; //nº de features (nº de linhas da matriz de entrada)
    //* 9.1. avalia o modelo treinado se o modo de avaliação está on (= 1)
    if (x->trainingmode_c == 0 && x->evalmode == 1){
        if (x->current_data < x->datasize) { //verifica se o índice do exemplo de teste é menor que o nº de exemplos de teste
            evaluation(x, n); //calcula as métricas de avaliação do model
            x->current_data++; //incrementa o índice do exemplo de teste
        }       
    
        if (x->current_data == x->datasize) { //se for o último exemplo de teste
            x->erro_total /= x->datasize; //média do erro para todos os exemplos de teste
            post("Total mean error: %lf", x->erro_total); //imprime o erro total
        }
    }
}


//*----------------------------------- função de criação do objeto cnn2d ----------------------------------------------------------
//se o objeto não receber argumentos ao lado do nome do objeto, a função cnn2d_new é void 
static void *cnn2d_new(void) {
    t_cnn2d *x = (t_cnn2d *)pd_new(cnn2d_class);

    //TODO: IMPELEMENTAR O CLINTE DE TEXTO PARA RECEBER MATRIZ DE DADOS DE ENTRADA

    x->simetria = 1; //simetria da matriz de kernel (1 = simétrica, 0 = não simétrica)

    x->bias_kernel = NULL; //vetor de bias

    //*ENCODER
    //define os vetores como null para evitar ponteiros que apontam para um local de memória inválido ou desalocado
    x->input_size = NULL;
    x->input_padded = NULL;
    x->padding = NULL;
    x->convolv_matriz_size = NULL;
    x->kernels_size = NULL;
    x->pooling_size = NULL;
    x->pooling_matriz_size = NULL;
    x->stride_conv = NULL;
    x->stride_pool = NULL;

    //define o array de matrizes como null para evitar ponteiros que apontam para um local de memória inválido 
    x->input = NULL;
    x->kernels = NULL;
    x->kernel_rotated = NULL;
    x->convolution = NULL;
    x->pooling = NULL;
    x->delta = NULL;
    x->conv_grad = NULL;
    x->kernels_grad = NULL;
    x->m_kernel = NULL;
    x->v_kernel = NULL;
    
    //*DECODER
    //define os vetores como null para evitar ponteiros que apontam para um local de memória inválido ou desalocado
    x->Tinput_size = NULL;
    x->Tinput_padded = NULL;
    x->Tpadding = NULL;
    x->Tconv_size = NULL;
    x->Tkernels_size = NULL;
    x->Tstride_conv = NULL;

    //define o array de matrizes como null para evitar ponteiros que apontam para um local de memória inválido
    x->Tinput = NULL;
    x->Tkernel = NULL;
    x->Tkernel_rotated = NULL;
    x->Tconv = NULL;
    x->Tdelta = NULL;
    x->Tkernel_grad = NULL;
    x->m_Tkernel = NULL;
    x->v_Tkernel = NULL;

    //vetores de simbolos para armazenar as funções de ativação, método de pooling e função de erro
    x->activation_function_c = NULL;
    x->pooling_function = NULL;
    x->error_function = NULL;
    x->optimizer = NULL;

    //buffers de saída
    x->latent_out = NULL;
    x->matrix_out = NULL;
    x->epoch_out = NULL;
    x->error_out = NULL;

    //*----------------------------- Nº DE CAMADAS ---------------------------
    x->num_layers = 2; //nº de camadas convolucionais do encoder (o calculo e alocação de todas matrizes e vetores depende do número de camadas)
    x->num_Tlayers = 2; //nº de camadas convolucionais do decoder
    x->learn_rate = 0.001; //taxa de aprendizado
    x->random_state = 42; //estado aleatório
    

    //*------------------------------- LIBERAÇÃO E ALOCAÇÃO DOS VETORES DE DIMENSÕES DAS MATRIZES ------------------------------- 

    //*ENCODER
    // Vetores relacionados ao número de camadas
    if (!ajustar_vetor(x, (t_float**)&x->input_size, 2, "Error allocating memory for the inpt data matrix size vector ") ||
        !ajustar_vetor(x, (t_float**)&x->padding, x->num_layers, "Error allocating memory for the encoder padding vector") ||
        !ajustar_vetor(x, (t_float**)&x->input_padded, x->num_layers * 2, "Error allocating memory for the encoder input padded matrices size vector") ||
        !ajustar_vetor(x, (t_float**)&x->kernels_size, x->num_layers * 2, "Error allocating memory for the encoder kernels size vector") ||
        !ajustar_vetor(x, (t_float**)&x->convolv_matriz_size, x->num_layers * 2, "Error allocating memory for the encoder convolution matrices size vector") ||
        !ajustar_vetor(x, (t_float**)&x->pooling_size, x->num_layers * 2, "Error allocating memory for the encoder pooling window size vector") ||
        !ajustar_vetor(x, (t_float**)&x->stride_conv, x->num_layers * 2, "Error allocating memory for the encoder convolution stride size vector") ||
        !ajustar_vetor(x, (t_float**)&x->stride_pool, x->num_layers * 2, "Error allocating memory for the encoder pooling stride size vector") ||
        !ajustar_vetor(x, (t_float**)&x->pooling_matriz_size, x->num_layers * 2, "Error allocating memory for the encoder pooling matrices size vector")) {
        return NULL; // Interrompe se houver erro
    }

    //*DECODER
    // Vetores relacionados ao número de camadas do decoder
    if (!ajustar_vetor(x, (t_float**)&x->Tinput_size, 2, "Error allocating memory for the decoder data matrix size vector") ||
        !ajustar_vetor(x, (t_float**)&x->Tpadding, x->num_Tlayers, "Error allocating memory for the decoder padding vector") ||
        !ajustar_vetor(x, (t_float**)&x->Tinput_padded, x->num_Tlayers * 2, "Error allocating memory for the decoder input padded matrices size vector") ||
        !ajustar_vetor(x, (t_float**)&x->Tkernels_size, x->num_Tlayers * 2, "Error allocating memory for the decoder kernels size vector") ||
        !ajustar_vetor(x, (t_float**)&x->Tconv_size, x->num_Tlayers * 2, "Error allocating memory for the decoder convolution matrices size vector") ||
        !ajustar_vetor(x, (t_float**)&x->Tstride_conv, x->num_Tlayers * 2, "Error allocating memory for the decoder convolution stride size vector")) {
        return NULL; // Interrompe se houver erro
    }

    //*------------------------------- ENCODER --------------------------------------------
    //------------------------------- DIMENSÕES DO PADDING -------------------------------
    x->padding[0] = 1; //padding para a primeira camada
    x->padding[1] = 1; //padding para a segunda camada
    //------------------------------- DIMENSÕES DA MATRIZ DE DADOS --------------------------------
    x->input_size[0] = 8;//linhas da matriz de dados sem padding
    x->input_size[1] = 8;//colunas da matriz de dados sem padding
    //------------------------------- DIMENSÕES DA MATRIZ DE KERNEL -------------------------------
    x->kernels_size[0] = 2;//LINHA - CAMADA 1
    x->kernels_size[1] = 2;//COLUNA - CAMADA 1
    x->kernels_size[2] = 2;//LINHA - CAMADA 2
    x->kernels_size[3] = 2;//COLUNA - CAMADA 2
    //------------------------------- DIMENSÕES DA JANELA DE POOLING -------------------------------
    x->pooling_size[0] = 2; //LINHA  da janela de pooling da camada 1
    x->pooling_size[1] = 2;//COLUNA - da janela de pooling da camada 1
    x->pooling_size[2] = 2;//LINHA - da janela de pooling da camada 2
    x->pooling_size[3] = 2;//COLUNA - da janela de pooling da camada 2
    //------------------------------- DIMENSÕES DO STRIDE DA CONVOLUÇÃO ----------------------------------
    x->stride_conv[0] = 1;//LINHA - CAMADA 1
    x->stride_conv[1] = 1;//COLUNA - CAMADA 1
    x->stride_conv[2] = 1;//LINHA - CAMADA 2
    x->stride_conv[3] = 1;//COLUNA - CAMADA2
    //----------------------------- DIMENSÕES DO STRIDE DO POOLING ---------------------------------
    x->stride_pool[0] = 1;//LINHA - CAMADA 1
    x->stride_pool[1] = 1;//COLUNA - CAMADA 1
    x->stride_pool[2] = 1;//LINHA - CAMADA 2
    x->stride_pool[3] = 1;//COLUNA - CAMADA 2

    //*------------------------------- DECODER --------------------------------------------------
    //------------------------------- DIMENSÕES DO PADDING -------------------------------
    x->Tpadding[0] = 0; //padding para a primeira camada do decoder
    x->Tpadding[1] = 1; //padding para a segunda camada do decoder
    //------------------------------- DIMENSÕES DA MATRIZ DE KERNEL -------------------------------
    x->Tkernels_size[0] = 2;//LINHA - CAMADA 1
    x->Tkernels_size[1] = 2;//COLUNA - CAMADA 1
    x->Tkernels_size[2] = 2;//LINHA - CAMADA 2
    x->Tkernels_size[3] = 2;//COLUNA - CAMADA 2
    //------------------------------- DIMENSÕES DO STRIDE DA CONVOLUÇÃO ----------------------------------
    x->Tstride_conv[0] = 1;//LINHA - CAMADA 1
    x->Tstride_conv[1] = 1;//COLUNA - CAMADA 1
    x->Tstride_conv[2] = 1;//LINHA - CAMADA 2
    x->Tstride_conv[3] = 1;//COLUNA - CAMADA 2

    //*------------------------------ CALCULA O TAMANHO DAS MATRIZES DO ENCODER E DECODER, LIBERA E ALOCA MEMÓRIA --------------------------
    create_network(x); //cria a rede (calcula o tamanho das matrizes e aloca memória)

    //*------------------------------- LIBERA E ALOCA AS MATRIZES DE KERNEL DO ENCODER --------------------------------------------
    free_matrix(x, &x->kernels, x->num_layers, x->kernels_size); //libera a memória das matrizes de kernel
    if(!alocar_matrizes(x, &x->kernels, x->num_layers, x->kernels_size)){
        error("Error allocating memory for the encoder kernels matrices");
        return NULL;
    }
    free_matrix(x, &x->kernel_rotated, x->num_layers, x->kernels_size); //libera a memória das matrizes de kernel
    if(!alocar_matrizes(x, &x->kernel_rotated, x->num_layers, x->kernels_size)){
        error("Error allocating memory for the encoder rotated kernels matrices");
        return NULL;
    }
    //*------------------------------------- LIBERA E ALOCA AS MATRIZES DE GRADIENTE DOS KERNELS DO ENCODER --------------------------------------------
    free_matrix(x, &x->kernels_grad, x->num_layers, x->kernels_size); //libera a memória das matrizes de gradiente do kernel do encoder
    if(!alocar_matrizes(x, &x->kernels_grad, x->num_layers, x->kernels_size)){
        error("Error allocating memory for the encoder gradient kernels matrices");
        return NULL;
    }
    //*------------------------------- LIBERA E ALOCA AS MATRIZES DO PRIMEIRO MOMENTO DO KERNEL DO ENCODER --------------------------------------------
    free_matrix(x, &x->m_kernel, x->num_layers, x->kernels_size); //libera a memória das matrizes de gradiente do kernel do encoder
    if(!alocar_matrizes(x, &x->m_kernel, x->num_layers, x->kernels_size)){
        error("Error allocating memory for the encoder kernels first moment matrices");
        return NULL;
    }
    //*------------------------------- LIBERA E ALOCA AS MATRIZES DO SEGUNDO MOMENTO DO KERNEL DO ENCODER --------------------------------------------
    free_matrix(x, &x->v_kernel, x->num_layers, x->kernels_size); //libera a memória das matrizes de gradiente do kernel do encoder
    if(!alocar_matrizes(x, &x->v_kernel, x->num_layers, x->kernels_size)){
        error("Error allocating memory for the encoder kernels second moment matrices");
        return NULL;
    }
    //*------------------------------- PREENCHE AS MATRIZES DE KERNELS E AS MATRIZES DO PRIMEIRO E SEGUNDO MOMENTO DO KERNEL DO ENCODER --------------------------------------------
    unsigned int state = (unsigned int)x->random_state; //semente para a geração de números aleatórios
    //preenche as matrizes de kernel com valores aleatórios
    matriz_fill(x, x->kernels, x->num_layers, x->kernels_size, 0, 1, &state, gensym("Encoder kernel"));
    //preenche as matrizes do primeiro e segundo momento do kernel com zeros
    matriz_fill(x, x->m_kernel, x->num_layers, x->kernels_size, 0, 0, &state, gensym("Encoder first moment kernel"));
    matriz_fill(x, x->v_kernel, x->num_layers, x->kernels_size, 0, 0, &state, gensym("Encoder second moment kernel"));


    //*------------------------------- LIBERA E ALOCA AS MATRIZES DE KERNEL DO DECODER --------------------------------------------
    free_matrix(x, &x->Tkernel, x->num_Tlayers, x->Tkernels_size); //libera a memória das matrizes de kernel do decoder
    if(!alocar_matrizes(x, &x->Tkernel, x->num_Tlayers, x->Tkernels_size)){
        error("Error allocating memory for the decoder kernels matrices");
        return NULL;
    }
    free_matrix(x, &x->Tkernel_rotated, x->num_Tlayers, x->Tkernels_size); //libera a memória das matrizes de kernel do decoder
    if(!alocar_matrizes(x, &x->Tkernel_rotated, x->num_Tlayers, x->Tkernels_size)){
        error("Error allocating memory for the decoder kernels matrices");
        return NULL;
    }
    //*------------------------------------- LIBERA E ALOCA AS MATRIZES DE GRADIENTE DOS KERNELS DO DECODER --------------------------------------------
    free_matrix(x, &x->Tkernel_grad, x->num_Tlayers, x->Tkernels_size); //libera a memória das matrizes de gradiente dos kernels do decoder
    if(!alocar_matrizes(x, &x->Tkernel_grad, x->num_Tlayers, x->Tkernels_size)){
        error("Error allocating memory for the decoder gradient matrices");
        return NULL;
    }
    //*------------------------------- LIBERA E ALOCA AS MATRIZES DO PRIMEIRO MOMENTO DO KERNEL DO DECODER --------------------------------------------
    free_matrix(x, &x->m_Tkernel, x->num_Tlayers, x->Tkernels_size); //libera a memória das matrizes de gradiente do kernel do decoder
    if(!alocar_matrizes(x, &x->m_Tkernel, x->num_Tlayers, x->Tkernels_size)){
        error("Error allocating memory for the decoder kernels first moment matrices");
        return NULL;
    }
    //*------------------------------- LIBERA E ALOCA AS MATRIZES DO SEGUNDO MOMENTO DO KERNEL DO DECODER --------------------------------------------
    free_matrix(x, &x->v_Tkernel, x->num_Tlayers, x->Tkernels_size); //libera a memória das matrizes de gradiente do kernel do decoder
    if(!alocar_matrizes(x, &x->v_Tkernel, x->num_Tlayers, x->Tkernels_size)){
        error("Error allocating memory for the decoder kernels second moment matrices");
        return NULL;
    }
    //*------------------------------- PREENCHE AS MATRIZES DE KERNELS E AS MATRIZES DO PRIMEIRO E SEGUNDO MOMENTO DO KERNEL DO DECODER --------------------------------------------
    // unsigned int state2 = (unsigned int)x->random_state; //semente para a geração de números aleatórios
    //preenche as matrizes de kernel do decoder com valores aleatórios
    matriz_fill(x, x->Tkernel, x->num_Tlayers, x->Tkernels_size, 0, 1, &state, gensym("Decoder kernel"));
    //preenche as matrizes do primeiro e segundo momento do kernel do decoder com zeros
    matriz_fill(x, x->m_Tkernel, x->num_Tlayers, x->Tkernels_size, 0, 0, &state, gensym("Decoder first moment kernel"));
    matriz_fill(x, x->v_Tkernel, x->num_Tlayers, x->Tkernels_size, 0, 0, &state, gensym("Decoder second moment kernel"));

    
    //*------------------------------- LIBERA E ALOCA VETOR DE FUNÇÃO DE ATIVAÇÃO DE CADA CAMADA (ENCODER E DECODER ) --------------------------------------------
    int totallayers = x->num_layers + x->num_Tlayers; //nº total de camadas (encoder + decoder)
    
    //libera a memória do array de funções de ativação
    if (x->activation_function_c != NULL) {
        freebytes(x->activation_function_c, totallayers * sizeof(t_symbol *)); // Libera a memória do array de funções de ativação
        x->activation_function_c = NULL;
    }
    //aloca memória para o array de funções de ativação
    x->activation_function_c = getbytes(totallayers * sizeof(t_symbol *)); // Aloca memória para o array de funções de ativação
    //preenche o array de funções de ativação com a função de ativação padrão
    for (int i = 0; i < totallayers; i++) {
        x->activation_function_c[i] = gensym("relu");  // Função de ativação padrão para todas as camadas
        post("Layer %d: activation function set to %s", i+1, x->activation_function_c[i]->s_name);
    }

    //*------------------------------- LIBERA E ALOCA VETOR DE MÉTODO DE POOLING PARA CADA CAMADA DO ENCODER --------------------------------------------
    //libera a memória do array de funções de pooling
    if (x->pooling_function != NULL) {
        freebytes(x->pooling_function, x->num_layers * sizeof(t_symbol *)); // Libera a memória do array de funções de pooling
        x->pooling_function = NULL;
    }
    //aloca memória para o array de funções de pooling
    x->pooling_function = getbytes(x->num_layers * sizeof(t_symbol *)); // Aloca memória para o array de funções de pooling
    //preenche o array de funções de pooling com a função de pooling padrão
    for (int i = 0; i < x->num_layers; i++) {
        x->pooling_function[i] = gensym("max"); // Função de pooling padrão para todas as camadas
        post("Layer %d: pooling function set to %s", i+1, x->pooling_function[i]->s_name);
    }

    //*------------------------------------ LIBERA E ALOCA VETOR DE NOME DA FUNÇÃO DE PERDA ---------------------------------
    //libera a memória do vetor de nome da função de perda
    if (x->error_function != NULL) {
        freebytes(x->error_function, sizeof(t_symbol *)); // Libera a memória do vetor de nome da função de perda
        x->error_function = NULL;
    }
    //aloca memória para o vetor de nome da função de perda
    x->error_function = getbytes(1 * sizeof(t_symbol *)); // Aloca memória para o vetor de nome da função de perda
    x->error_function = gensym("mse"); // Função de perda padrão
    post("Loss function set to %s", x->error_function->s_name);

    //*------------------------------------ LIBERA E ALOCA VETOR DE NOME DO OTIMIZADOR ---------------------------------
    //libera a memória do vetor de nome do otimizador
    if (x->optimizer != NULL) {
        freebytes(x->optimizer, sizeof(t_symbol *)); // Libera a memória do vetor de nome da função de perda
        x->optimizer = NULL;
    }
    //aloca memória para o vetor de nome do otimizador
    x->optimizer = getbytes(1 * sizeof(t_symbol *)); // Aloca memória para o vetor de nome da função de perda
    x->optimizer = gensym("sgd"); // otimizador padrão
    post("Optimizer set to %s", x->optimizer->s_name);


    //*------------------------------------ ALOCA BUFFER DE SAÍDA DO ERRO ---------------------------------
    //libera se o buffer de saída do erro já foi alocado
    if (x->error_out != NULL) {    
        freebytes(x->error_out, 1 * sizeof(t_atom)); // Libera a memória do buffer de saída do espaço latente (nº de colunas da última camada do encoder)
        x->error_out = NULL;
    }
    //aloca memória para o buffer de saída do erro
    x->error_out = (t_atom *)getbytes(1 * sizeof(t_atom)); 
    if (x->error_out == NULL) {
        error("Error allocating memory for the error output buffer");
        return NULL;
    }

    //*------------------------------------ ALOCA BUFFER DE SAÍDA DA ÉPOCA ---------------------------------
    //libera se o buffer de saída da época já foi alocado
    if (x->epoch_out != NULL) {    
        freebytes(x->epoch_out, 1 * sizeof(t_atom)); // Libera a memória do buffer de saída do espaço latente (nº de colunas da última camada do encoder)
        x->epoch_out = NULL;
    }
    //aloca memória para o buffer de saída da época
    x->epoch_out = (t_atom *)getbytes(1 * sizeof(t_atom)); 
    if (x->epoch_out == NULL) {
        error("Error allocating memory for the epoch output buffer");
        return NULL;
    }

    //verifica alocação de memória do vetor de bias
    if (x->bias_kernel == NULL) {
        error("BIAS NÃO FOI ALOCADO DENTRO DE NEW");
    }

    post("cnn2d v0.01 created successfully");

    //* ------------------------- cria os outlets -------------------------------------
    x->x_out1 = outlet_new(&x->x_obj, &s_list); //espaço latente
    x->x_out2 = outlet_new(&x->x_obj, &s_list); //matriz reconstruída
    x->x_out3 = outlet_new(&x->x_obj, &s_anything); //erro e épocas
    return (void *)x;
}

//*---------------------------------- função de destruição do objeto cnn2d -------------------------------------
static void cnn2d_destroy(t_cnn2d *x) {
    outlet_free(x->x_out1); // Libera o outlet 1
    outlet_free(x->x_out2);// Libera o outlet 2
    outlet_free(x->x_out3);// Libera o outlet 3

    //liberar memória do vetor de bias
    free_bias(x, &x->bias_kernel);   
    //liberar memória do vetor do primeiro momento do bias
    free_momtento_bias(x, &x->m_bias);
    //liberar memória do vetor do segundo momento do bias
    free_momtento_bias(x, &x->v_bias);

    //*ENCODER
    //matrizes do encoder
    free_matrix(x, &x->input, x->num_layers, x->input_padded);//libera matrizes de entrada -- uma matriz para cada camada (matriz de dados + matriz de pooling da camada anterior)
    free_matrix(x, &x->kernels, x->num_layers, x->kernels_size);//libera matrizes de kernel
    free_matrix(x, &x->kernel_rotated, x->num_layers, x->kernels_size);//libera matrizes de kernel rotacionado
    free_matrix(x, &x->convolution, x->num_layers, x->convolv_matriz_size);//libera matrizes de convolução
    free_matrix(x, &x->pooling, x->num_layers, x->pooling_matriz_size);//libera matrizes de pooling
    //matrizes de delta do encoder
    free_matrix(x, &x->delta, x->num_layers, x->pooling_matriz_size);//libera matrizes de delta das saídas de cada camada
    free_matrix(x, &x->kernels_grad, x->num_layers, x->kernels_size);//libera matrizes de gradiente do kernel
    free_matrix(x, &x->conv_grad, x->num_layers, x->convolv_matriz_size);//libera matrizes de gradiente da convolução
    //matrizes do primeiro e segundo momento do kernel
    free_matrix(x, &x->m_kernel, x->num_layers, x->kernels_size);//libera matrizes do primeiro momento do kernel
    free_matrix(x, &x->v_kernel, x->num_layers, x->kernels_size);//libera matrizes do segundo momento do kernel

    //*DECODER
    //matrizes do decoder
    free_matrix(x, &x->Tinput, x->num_Tlayers, x->Tinput_padded);//libera matrizes de entrada do decoder
    free_matrix(x, &x->Tkernel, x->num_Tlayers, x->Tkernels_size);//libera matrizes de kernel do decoder
    free_matrix(x, &x->Tkernel_rotated, x->num_Tlayers, x->Tkernels_size);//libera matrizes de kernel do decoder
    free_matrix(x, &x->Tconv, x->num_Tlayers, x->Tconv_size);//libera matrizes de convolução do decoder
    //matrizes de delta do decoder
    free_matrix(x, &x->Tdelta, x->num_Tlayers, x->Tconv_size);//libera matrizes de delta do decoder
    free_matrix(x, &x->Tkernel_grad, x->num_Tlayers, x->Tkernels_size);//libera matrizes de kernel do decoder
    //matrizes do primeiro e segundo momento do kernel do decoder
    free_matrix(x, &x->m_Tkernel, x->num_Tlayers, x->Tkernels_size);//libera matrizes do primeiro momento do kernel do decoder
    free_matrix(x, &x->v_Tkernel, x->num_Tlayers, x->Tkernels_size);//libera matrizes do segundo momento do kernel do decoder
    
    //*VETORES DE DIMENSÕES DAS MATRIZES DO ENCODER
    liberar_vetor(x, &x->input_size);//libera vetor com dimensões da matriz de dados
    liberar_vetor(x, &x->input_padded);//libera vetor com dimensões da matriz de entrada de cada camada
    liberar_vetor(x, &x->kernels_size);//libera vetor com dimensões dos kernels
    liberar_vetor(x, &x->pooling_size); //libera vetor com dimensões dos pooling
    liberar_vetor(x, &x->stride_conv);//libera vetor com valores de stride da convolução
    liberar_vetor(x, &x->stride_pool);//libera vetor com valores de stride dos poolings
    liberar_vetor(x, &x->padding);//libera vetor com valor de padding
    liberar_vetor(x, &x->convolv_matriz_size);//libera vetor com dimensões das matrizes de convolução
    liberar_vetor(x, &x->pooling_matriz_size);//libera vetor com dimensões das matrizes de pooling

    //*VETORES DE DIMENSÕES DAS MATRIZES DO DECODER
    liberar_vetor(x, &x->Tinput_size);//libera vetor com dimensões da matriz de entrada do decoder
    liberar_vetor(x, &x->Tpadding);//libera vetor com dimensões do padding do decoder
    liberar_vetor(x, &x->Tinput_padded);//libera vetor com dimensões da matriz de entrada do decoder
    liberar_vetor(x, &x->Tkernels_size);//libera vetor com dimensões dos kernels do decoder
    liberar_vetor(x, &x->Tconv_size);//libera vetor com dimensões das matrizes de convolução do decoder
    liberar_vetor(x, &x->Tstride_conv);//libera vetor com dimensões do stride de convolução do decoder

    //*VETORES DE SIMBOLOS   
    //libera memória do array de funções de ativação
    int total_layers = x->num_layers + x->num_Tlayers;
    if (x->activation_function_c != NULL) {
        freebytes(x->activation_function_c, total_layers * sizeof(t_symbol *)); // Libera a memória do array de funções de ativação
        x->activation_function_c = NULL;
    }

    //libera memória do array de método de pooling
    if (x->pooling_function != NULL) {
        freebytes(x->pooling_function, x->num_layers * sizeof(t_symbol *)); // Libera a memória do array de funções de pooling
        x->pooling_function = NULL;
    }

    //libera memória do vetor de nome da função de perda
    if (x->error_function != NULL) {
        freebytes(x->error_function, sizeof(t_symbol *)); // Libera a memória do vetor de nome da função de perda
        x->error_function = NULL;
    }

    //libera memória do vetor de nome do otimizador
    if (x->optimizer != NULL) {
        freebytes(x->optimizer, sizeof(t_symbol *)); // Libera a memória do vetor de nome da função de perda
        x->optimizer = NULL;
    }

    //* BUFFERS DE SAÍDA
    if (x->latent_out != NULL) {
        freebytes(x->latent_out, sizeof(t_atom)); // Libera a memória do buffer de saída
        x->latent_out = NULL;
    }

    //BUFFER DE SAÍDA da matriz reconstruída
    if (x->matrix_out != NULL) {
        freebytes(x->matrix_out, sizeof(t_atom)); // Libera a memória do buffer de saída
        x->matrix_out = NULL;
    }

    //BUFFER DE SAÍDA DO ERRO
    if (x->error_out != NULL) {
        freebytes(x->error_out, sizeof(t_atom)); // Libera a memória do buffer de saída
        x->error_out = NULL;
    }

    //BUFFER DE SAÍDA DA ÉPOCA
    if (x->epoch_out != NULL) {
        freebytes(x->epoch_out, sizeof(t_atom)); // Libera a memória do buffer de saída
        x->epoch_out = NULL;
    }
}

//*---------------------------------- função de setup do objeto cnn2d -------------------------------------
void cnn2d_setup(void) {
    cnn2d_class = class_new(
        gensym("cnn2d"), // Nome do objeto
        (t_newmethod)cnn2d_new, // Chama a função construtor
        (t_method)cnn2d_destroy, // Chama a função destruidor 
        sizeof(t_cnn2d),
        CLASS_DEFAULT,
        A_DEFFLOAT, 0); // Tamanho do objeto    

    class_addlist(cnn2d_class, (t_method) input_data_prop); //recebe lista com valores que serão atribuídos à matriz de entrada da primeira camada e realiza convolução e pooling
    class_addmethod(cnn2d_class, (t_method)input_data_size, gensym("inputsize"), A_GIMME, 0); //recebe lista com pares de linhas e colunas para matriz de entrada
    class_addmethod(cnn2d_class, (t_method)num_camadas, gensym("layers"), A_GIMME, 0); //recebe o número de camadas (1 valor para simetria = 1 e 2 valores para simetria = 0)
    class_addmethod(cnn2d_class, (t_method)symmetry_mode, gensym("symmetry"), A_FLOAT, 0); //recebe o valor de simetria do encoder e decoder (1 = simétrica, 0 = não simétrica)
    class_addmethod(cnn2d_class, (t_method)kernels_size, gensym("kernels"), A_GIMME, 0); //recebe lista com pares de linhas e colunas para cada kernel
    class_addmethod(cnn2d_class, (t_method)pooling_size, gensym("poolingwin"), A_GIMME, 0); //recebe lista com pares de linhas e colunas de cada janela de pooling
    class_addmethod(cnn2d_class, (t_method)stride_convolution, gensym("convstride"), A_GIMME, 0); //recebe lista com valores de stride para cada convolução
    class_addmethod(cnn2d_class, (t_method)stride_pooling, gensym("poolstride"), A_GIMME, 0); //recebe lista com valores de sride para cada pooling
    class_addmethod(cnn2d_class, (t_method)padding, gensym("padding"), A_GIMME, 0); //recebe uma lista com os valores de padding de cada camada
    class_addmethod(cnn2d_class, (t_method)create_network, gensym("create"), A_GIMME, 0);//cria a rede a partir dos parâmetros fornecidos (libera, aloca e atribui dimensões das matrizes)
    class_addmethod(cnn2d_class, (t_method)decoder_padding, gensym("decoderPadding"), A_GIMME, 0); //recebe uma lista com os valores de padding de cada camada do decoder (somente se simetria = 0)
    class_addmethod(cnn2d_class, (t_method)decoder_kernel_size, gensym("decoderKernels"), A_GIMME, 0); //recebe lista com pares de linhas e colunas para cada kernel do decoder (somente se simetria = 0)
    class_addmethod(cnn2d_class, (t_method)decoder_stride, gensym("decoderStride"), A_GIMME, 0); //recebe lista com valores de stride para cada convolução do decoder (somente se simetria = 0)
    class_addmethod(cnn2d_class, (t_method)activation_functions, gensym("activation"), A_GIMME, 0); //define as funções de ativação para cada camada do encoder e decoder
    class_addmethod(cnn2d_class, (t_method)pooling_methods, gensym("poolingMethod"), A_GIMME, 0); //define os métodos de pooling para cada camada do encoder
    class_addmethod(cnn2d_class, (t_method)error_function, gensym("loss"), A_GIMME, 0); //recebe nome da função de custo
    class_addmethod(cnn2d_class, (t_method)training_mode, gensym("training"), A_FLOAT, 0); //define o modo de treinamento
    class_addmethod(cnn2d_class, (t_method)evalmode, gensym("evaluation"), A_FLOAT, 0); //define o modo de avaliação
    class_addmethod(cnn2d_class, (t_method)epoch_amount, gensym("epochs"), A_FLOAT, 0);//define o nº máximo de épocas
    class_addmethod(cnn2d_class, (t_method)training_examples, gensym("datasize"), A_FLOAT, 0);//define o nº de exemplos de treinamento
    class_addmethod(cnn2d_class, (t_method)optimizer, gensym("optimizer"), A_GIMME, 0); //recebe nome do otimizador para atualização dos pesos
    class_addmethod(cnn2d_class, (t_method)learning_rate, gensym("learning"), A_FLOAT, 0);//define a taxa de aprendizado
    class_addmethod(cnn2d_class, (t_method)random_he, gensym("he"), A_GIMME, 0); //preenche os vetores de bias e matriz de pesos com valores aleatórios he
    class_addmethod(cnn2d_class, (t_method)random_xavier, gensym("xavier"), A_GIMME, 0);//preenche os vetores de bias e matriz de pesos com valores aleatórios xavier
    class_addmethod(cnn2d_class, (t_method)random_state, gensym("randomstate"), A_FLOAT, 0); //define o modo de treinamento
}



