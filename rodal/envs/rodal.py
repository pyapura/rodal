# -*- coding: utf-8 -*-
import gym
from gym import spaces
from gym.utils import seeding
import math

RCG = 6                 #Rango_clase_G
RCE = 5                 #Rango_clase_E
S = 20                  #Índice de sitio
s = 0.95                #Supervivencia
EI = 4                  #Edad inicial
PER = 5                 #Primera edad de raleo
#PEC = 15                #Primera edad de cosecha (actualmente no implementada)
UEC = 30                #Última edad de cosecha (turno máximo)
densidades_iniciales = [625, 816, 1111, 1600, 2500] # como un mazo de naipes

#GEI: área basal a la edad inicial; NP: densidad de plantación
def crecer_GEI(NP, S = 20, EI = 4, s = 0.95):
    return math.exp(-4.701985 - 0.353551*S/EI + 0.596669*math.log(NP*s) + 0.220277*S)
#G2: área basal al final de un período anual de crecimiento;
#E2: edad al final del período; G1: área basal al inicio del período 
def crecer_G2(E2, G1, S = 20):
    return math.exp((E2-1)/E2*math.log(G1) + 2.098464*(1-(E2-1)/E2) + 0.096649*S*(1-(E2-1)/E2))
#V: volumen total con corteza
def calcular_V(E, G, S = 20):
    return math.exp(1.627362 + 0.058201*S - 6.937454/E + 0.949623*math.log(G))
#clase_G: índice (base 0) de la clase de área basal
def asignar_clase_G(G, rango_clase_G):
    return math.floor(G/rango_clase_G)
#mc_G: marca central de la clase de área basal
def calcular_mc_G(indice_clase_G, rango_clase_G):
    return indice_clase_G*rango_clase_G + rango_clase_G/2
#clase_E: índice (base 0) de la clase de edad
def asignar_clase_E(E, rango_clase_E):
    return int(E/rango_clase_E - 1) #if E > 0 else 0
#mc_E: marca de la clase de edad
def calcular_mc_E(indice_clase_E, rango_clase_E):
    return indice_clase_E*rango_clase_E + rango_clase_E
#estado: área basal y edad después de la acción (raleo) y el crecimiento
def actualizar_estado(area_basal, edad, accion):
    if accion == 1:
        area_basal *= 1 - 0.15
    elif accion == 2:
        area_basal *= 1 - 0.30
    for edad in range(edad, edad+RCE):
        area_basal = crecer_G2(edad+1, area_basal)
    return area_basal, edad+1
#Recompensa: por ahora el volumen total cosechado
def calcular_recompensa(area_basal, edad, accion):
    V_antes = calcular_V(edad, area_basal)
    if accion == 1:
        area_basal *= 1 - 0.15
        V_despues = calcular_V(edad, area_basal)
    elif accion == 2:
        area_basal *= 1 - 0.30
        V_despues = calcular_V(edad, area_basal)
    elif accion == 3:
        return V_antes
    return V_antes - V_despues

class PtDiscreto6x5(gym.Env):
    """
    Descripción:
        Se establece un rodal de Pinus taeda con una densidad de plantación dada. El objetivo es determinar la oportunidad e intensidad de los raleos para maximizar la producción de madera a lo largo de toda la rotación.
    
    Espacio de estados u Observaciones:
        Tipo: Tuple(Discrete(10), Discrete(6))
        Discrete(10): área basal del rodal, actualmente expresada en 10 clases de 6 m^2/ha de amplitud, i.e. los índices se mapean al conjunto discreto G = {3, 9, 15, 21, 27, 33, 39, 44, 50, 56}
        Discrete(6): edad del rodal, actualmente expresada en períodos de 5 años, i.e. los índices se mapean con el conjunto discreto E = {5, 10, 15, 20, 25, 30)
        
    Acciones:
        Tipo: Discrete(4)
        Índice    Descripción de la acción
        0         No ralear ni cosechar el rodal
        1         Ralear el 15% del área basal del rodal
        2         Ralear el 30% del área basal del rodal
        3         Cosechar el rodal
        Actualmente, todas las acciones están permitidas en todas las edades, excepto a los 30 años en los que sólo se permite la cosecha.
        Nota: para que el espacio de acciones cambie con la edad, consultar: https://stackoverflow.com/questions/45001361/open-ai-enviroment-with-changing-action-space-after-each-step
    
    Recompensa:
        La recompensa es el volumen total raleado o cosechado, según corresponda y en m^3/ha.

    Estado inicial:
        La densidad de plantación (número de árboles) se fija aleatoriamente entre cinco valores prefijados: 625, 816, 1111, 1600, 2500 árboles/ha, los que se corresponden aproximadamente con los siguientes distanciamientos en m: 4x4; 3,5x3,5; 3x3, 2,5x2,5 y 2x2. Al número de árboles plantados seleccionado se la detrae una mortalidad (actualmente 5%), la que corresponde al período que va desde el establecimiento hasta una edad inicial (actualmente 4 años). En este momento la densidad de plantación es convertida a área basal con una función que depende del índice de sitio y se proyecta su crecimiento hasta la primera edad de raleo (actualmente 5 años). El espacio de estados inicial está constituido, en consecuencia, por cinco observaciones del área basal, todas a los 5 años de edad.

    Terminación de un episodio:
        Los episodios terminan cuando se cosecha el rodal, lo cual puede ocurrir en todas las edades, i.e. a los 5 años como mínimo, y a los 30 años como máximo.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(10), spaces.Discrete(6)))
        self.observation = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def reset(self):
        NP = self.np_random.choice(densidades_iniciales)
        G_inicial = crecer_G2(PER, crecer_GEI(NP))
        self.observation = self.set_observation(G_inicial, PER)
        return self.observation

    def step(self, action):
        assert self.action_space.contains(action)
        area_basal, edad = self.get_observation(self.observation)
        if edad == UEC: action = 3
        if action == 0:
            done = False
            reward = 0
            area_basal, edad = actualizar_estado(area_basal, edad, action)
        elif action != 3:
            done = False
            reward = calcular_recompensa(area_basal, edad, action)
            area_basal, edad = actualizar_estado(area_basal, edad, action)
        elif action == 3:
            done = True
            reward = calcular_recompensa(area_basal, edad, action)
            area_basal, edad = 0, 0
        self.observation = self.set_observation(area_basal, edad)
        return self.observation, reward, done, {}
    
    def set_observation(self, area_basal, edad):
        return (asignar_clase_G(area_basal, RCG), asignar_clase_E(edad, RCE))

    def get_observation(self, observation):
        return calcular_mc_G(observation[0], RCG), calcular_mc_E(observation[1], RCE)

    def render(self): pass
