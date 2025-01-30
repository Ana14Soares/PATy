import cv2
import mediapipe as mp
import subprocess
from time import sleep 
from pynput.keyboard import Controller

#salvando como constante as cores
BRANCO = (255,255,255)
PRETO = (0,0,0)
AZUL = (255,0,0)
VERDE = (0, 255, 0)
VERMELHO = (0,0,255)
AZUL_CLARO = (255,255,0)
offset = 70
teclas = [['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'], ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K','L','Ç'],['Z','X','C','V','B','N','M','.',',',' ']]


#vamos acessar duas soluções da biblioteca
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

#adicionar os modelos de machine learning que fazem 
#a detecção
maos = mp_maos.Hands()

camera = cv2.VideoCapture(0)
resolucao_x = 1280
resolucao_y = 720

#evitar que os apps se abram em loop infinito
chatGPT = False
instagram = False
whatsWeb = False

#variaveis para fazer o contador para o texto
contador = 0
texto = '> '

#resolucao da camera
camera.set(cv2.CAP_PROP_FRAME_WIDTH,resolucao_x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucao_y)


#fazendo função para identificar se o dedo ta levantado ou nao
def dedos_levantados(mao):

    #lista para se o dedo ta levantado (True) ou nao (False)
    dedos = []

#verificando o dedao horizontalmente, pois pode ser que ele nn esteja totalmente na veritical 
    if mao['Lado'] == 'Right': 
        if mao['coordenadas'][4][0] < mao['coordenadas'][3][0]:
            dedos.append(True)
        else:
            dedos.append(False)
    else:
        if mao['coordenadas'][4][0] > mao['coordenadas'][3][0]:
            dedos.append(True)
        else:
            dedos.append(False)

    #verificando se o dedo esta levantado ou nao de acordo com os pontos da mao, o da ponta do dedo e o mais baixo
    for ponta_dedo in [8, 12, 16, 20]:
        if mao['coordenadas'][ponta_dedo][1] < mao['coordenadas'][ponta_dedo - 2][1]:
            dedos.append(True)
        else:
            dedos.append(False)

    return dedos             

#função para imprimir os botões
def imprime_botoes(img, posicao, letra, tamanho = 50, cor_retangulo = BRANCO):
     
    cv2.rectangle(img, posicao, (posicao[0] + tamanho, posicao[1] + tamanho), cor_retangulo, cv2.FILLED)
    cv2.rectangle(img, posicao, (posicao[0] + tamanho, posicao[1] + tamanho), AZUL, 1)

    #usar a função puText para escrever o texto na tecla 

    cv2.putText(img, letra, (posicao[0] + 15, posicao[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, PRETO, 2)
    return img

def encontra_coordenadas_maos(img, lado_invertido = False):
     #o mediapipe so processam imagens em rbg, mas o openCV
    #usa imagens em brg, por isso vamos fazer uma troca
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #processar a imagem com o modelo de machine learning
    resultado = maos.process(img_rgb)

    #vamos usar o resultado multi_hands_landmarks do processamento

    #criando uma lista para armazenar um dicionario que vai ter as informações das maõs
    todas_maos = []

    #checando se tem mãos na tela
    if resultado.multi_hand_landmarks:
        #desenhar em cada marcação detectada
        for lado_mao, marcacao_maos in zip(resultado.multi_handedness, resultado.multi_hand_landmarks):

            #dicionario para asmazenar informações
            info_maos = {}

            #lista para armazenar as coordenadas
            coordenadas = []
            #acessar as coordenadas de cada landmark e transformar em pixels
            for marcacao in marcacao_maos.landmark:
                cord_x, cord_y, cord_z = int(marcacao.x * resolucao_x), int(marcacao.y * resolucao_y), int(marcacao.z * resolucao_x)
                
                #usando uma tupla
                coordenadas.append((cord_x, cord_y, cord_z ))
            
            #armazenando as coordenadas no dicionario 
            info_maos['coordenadas'] = coordenadas

            #criando uma logia para inverter os lados, se necessario
            #se lado invertido for True, vai entrar no if
            if lado_invertido:
                if lado_mao.classification[0].label == "Left":
                    info_maos["Lado"] = "Right"
                else:
                    info_maos["Lado"] = "Left"
            else:
                info_maos["Lado"] = lado_mao.classification[0].label            


            #armazenando o dicionario na lista
            todas_maos.append(info_maos)

             #desenhe
            mp_desenho.draw_landmarks(img,marcacao_maos,mp_maos.HAND_CONNECTIONS)
    return img, todas_maos

while True:
    #leitura das imagens com a função read
    sucesso,img = camera.read()

    #invertendo o lado das imagens
    img = cv2.flip(img, 1)

    img, todas_maos = encontra_coordenadas_maos(img)

    #identificando se so tem uma mao na tela
    if len(todas_maos) == 1:
        #criando uma variavel para armazenar os dedos levantados
        info_dedos_mao1 = dedos_levantados(todas_maos[0])
        if todas_maos[0]['Lado'] == 'Left':
            #achar as coordenadas 
            indicador_x, indicador_y, indicador_z = todas_maos[0]['coordenadas'][8]
            #escrever a distancia z na tela
            cv2.putText(img, f'Distancia Camera {indicador_z}', (850, 50), cv2.FONT_HERSHEY_COMPLEX, 1, BRANCO, 2)
            #ler as linhas e colunas da lista
            for indice_linha, linha_teclado in enumerate(teclas):
                for indice, letra in enumerate(linha_teclado):
                    #contar os dedos na tela
                    if sum(info_dedos_mao1) <= 1:
                        #transformar em letra minuscula
                        letra = letra.lower()
                    img = imprime_botoes(img, (offset + indice*80, offset + indice_linha*80), letra)
                    #checar se o dedo esta na tecla e ficar verde
                    if offset+ indice * 80 < indicador_x < 100 + indice * 80 and offset + indice_linha * 80 < indicador_y < 100 + indice_linha * 80:
                        img = imprime_botoes(img, (offset + indice*80, offset + indice_linha*80), letra, cor_retangulo = VERDE)
                        #checar a distancia do dedo no Z
                        if indicador_z < -85:
                            contador = 1
                            escreve = letra
                            img = imprime_botoes(img, (offset + indice*80, offset + indice_linha*80), letra, cor_retangulo = AZUL_CLARO)

            #contador para as letras                
            if contador:
                contador += 1
                if contador == 3:
                    texto += escreve 
                    contador = 0

            #para apagar o texto
            if info_dedos_mao1 == [False, False, False, False, True] and len(texto) > 1:
                texto = texto[:-1]
                sleep(0.25)        

            #mostrar na tela 
            cv2.rectangle(img, (offset, 450), (830, 500), BRANCO, cv2.FILLED)
            cv2.rectangle(img, (offset, 450), (830, 500), AZUL, 1)
            cv2.putText(img, texto[-40:], (offset, 480), cv2.FONT_HERSHEY_COMPLEX, 1, PRETO, 2 )
            cv2.circle(img, (indicador_x, indicador_y), 7, AZUL, cv2.FILLED)




        if todas_maos[0]['Lado'] == 'Right':
            #verificar se o indicador esta levantado 
            if info_dedos_mao1 == [False, True, False, False, False] and instagram == False:
                #o startfile nao funciona no linux, entao vamos substituir 
                subprocess.run(['xdg-open', 'https://www.instagram.com/'])
                instagram = True
            if info_dedos_mao1 == [False, False, True, False, False] and chatGPT == False:
                #o startfile nao funciona no linux, entao vamos substituir 
                subprocess.run(['xdg-open', 'https://chatgpt.com/'])
                chatGPT = True
            if info_dedos_mao1 == [False, True, True, True, False] and whatsWeb == False:
                #o startfile nao funciona no linux, entao vamos substituir 
                subprocess.run(['xdg-open', 'https://web.whatsapp.com'])
                whatsWeb = True
    
    

    #mostrar a imagem na tela com a função imshow
    cv2.imshow("imagem",img)

    #pegar frames dom i milissegundo de pausa

    tecla = cv2.waitKey(1)

    #apertar tecla esc para encerrar o loop
    if tecla == 27:
        break

    with open('texto.txt', 'w') as arquivo:
        arquivo.write(texto)
