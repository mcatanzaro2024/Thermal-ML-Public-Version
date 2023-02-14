import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import GridSimulator

import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.env_checker import check_env

import random

# This environment/agent is designed to operate in (x3) simulations
#  (1) Hot cycle
#  (2) Cold cycle
#  (3) Sustain cycle
#  The agent can control the temperature and duration of each of the first (x2) cycles

#  The agent will have access to a few of the temperatures of the system to help it learn
#  The reward will be based upon:
#    +fraction for the amount of time that the last node of the pan is in the acceptable range of temperatures to cook
#    -fraction for the amount of time that the last node of the pan is burning pancakes
#    0 for the amount of time that the pan is too cold

class EnvAgentThreeStep(Env):

    # Restrict the temperature availalbe

    listTemperatureHot = (200.0, 300.0, 500.0)
    listTemperatureCold = (50.0, 100.0, 135.0)
    listTimeHot = (60.0, 120.0, 240.0, 480.0)
    listTimeCold = (60.0, 120.0, 240.0, 480.0)

    temperatureSustain = 133

    # Configure simulation
    numberNodes = 10
    thicknessTotal = 0.1
    widthBar = 0.003
    thicknessBar = 0.003

    materialDensity = 7800
    materialSpecificHeat = 500
    materialThermalConductivity = 16.2

    convectionCoefficient = 4
    convectionTemperature = 25

    temperatureHeatSource = 25
    temperatureInitialIsothermal = 25
    doesSurfaceTemperatureAllowConvection = False

    # Number of Points on the griddle that return temperature information
    noPtsObserveTemperature = 6
    temperatureMinObservation = 0.0
    temperatureMaxObservation = 500.0

    timeTotal = 3600

    def __init__(self):
        self.action_space = MultiDiscrete([3, 4])
        self.observation_space = Box(\
            low=np.array(np.ones(self.noPtsObserveTemperature)*self.temperatureMinObservation, dtype=np.float32), \
            high=np.array(np.ones(self.noPtsObserveTemperature)*self.temperatureMaxObservation, dtype=np.float32))

        self.reset()

    def step(self, action):

        done = False
        if(self.currentStep==0):
            obs, reward, info = self.phaseHot(action)
            self.currentStep += 1
        elif(self.currentStep==1):
            obs, reward, info = self.phaseCold(action)
            self.currentStep += 1
            done = True

        return obs, reward, done, info

    def phaseHot(self, action):
        temperatureHeatSource = self.listTemperatureHot[action[0]]
        timeIncrement = self.listTimeHot[action[1]]

        timeStart = np.max(self.objHeatSimulator.AllTemp.Time)
        self.objHeatSimulator.AdvanceSimulationByTime(timeIncrement, temperatureHeatSource)
        timeEnd = np.max(self.objHeatSimulator.AllTemp.Time)

        isPhaseCurrent = (self.objHeatSimulator.AllTemp.Time > timeStart) & \
            (self.objHeatSimulator.AllTemp.Time < timeEnd)

        dfTemp = self.objHeatSimulator.AllTemp[isPhaseCurrent]
        reward, obs = self.computeRewardAndObservation(dfTemp)

        info = {"timeIncrement":timeIncrement, \
                "temperatureHeatSource":temperatureHeatSource,
                "timeCurrent":dfTemp.Time.to_numpy(), 
                "temperatureNode10":dfTemp.Node10.to_numpy()}

        return obs, reward, info

    def phaseCold(self, action):
        temperatureHeatSource = self.listTemperatureHot[action[0]]
        timeIncrement = self.listTimeHot[action[1]]

        timeStart = np.max(self.objHeatSimulator.AllTemp.Time)
        self.objHeatSimulator.AdvanceSimulationByTime(timeIncrement, temperatureHeatSource)
        timeEnd = np.max(self.objHeatSimulator.AllTemp.Time)

        timeSustain = self.timeTotal - timeEnd
        self.objHeatSimulator.AdvanceSimulationByTime(timeSustain, self.temperatureSustain)

        isPhaseCurrent = (self.objHeatSimulator.AllTemp.Time > timeStart)

        dfTemp = self.objHeatSimulator.AllTemp[isPhaseCurrent]
        reward, obs = self.computeRewardAndObservation(dfTemp)

        info = {"timeIncrement":timeIncrement, \
                "temperatureHeatSource":temperatureHeatSource,
                "timeCurrent":dfTemp.Time.to_numpy(), 
                "temperatureNode10":dfTemp.Node10.to_numpy()}

        return obs, reward, info

    def computeRewardAndObservation(self, dfTemp):
        timeStart = np.min(dfTemp.Time)
        timeEnd = np.max(dfTemp.Time)

        isTemperatureGoodForCooking = (dfTemp.Node10 > 130) & \
            (dfTemp.Node10 < 135)
      
        scoreOverTime = np.zeros(isTemperatureGoodForCooking.shape)
        scoreOverTime[isTemperatureGoodForCooking] = 1.0

        isTemperatureBurnt = (dfTemp.Node10 > 135)
        scoreOverTime[isTemperatureBurnt] = -1.0
        
        scoreIntegral = np.trapz(scoreOverTime, dfTemp.Time.to_numpy())
        score = scoreIntegral / np.max(dfTemp.Time)

        self.timeCoarse = np.linspace(timeStart, timeEnd, self.noPtsObserveTemperature)

        self.temperatureCoarse = np.interp(self.timeCoarse, dfTemp.Time, \
            dfTemp.Node10)

        reward = score
        listTemperature = self.temperatureCoarse
        obs = np.array(listTemperature, dtype=np.float32)

        return reward, obs

    def reset(self):

        obs = np.array(np.ones(self.noPtsObserveTemperature)*self.temperatureMinObservation, dtype=np.float32)
        self.createSimulator()
        self.currentStep = 0

        return obs

    def render(self):
        pass

    def createSimulator(self):

        self.objHeatSimulator = GridSimulator.ClassHeatSimulation()
        self.objHeatSimulator.SetSize(self.widthBar,self.thicknessBar)
        self.objHeatSimulator.SetNodes(self.numberNodes,self.thicknessTotal)
        self.objHeatSimulator.SetMaterialProperties(self.materialDensity,self.materialSpecificHeat,self.materialThermalConductivity)
        self.objHeatSimulator.SetConvectionProperties(self.convectionCoefficient,self.convectionTemperature)
        self.objHeatSimulator.EnableSurfaceHeatTransfer(self.doesSurfaceTemperatureAllowConvection)
        self.objHeatSimulator.SetHeatSource(self.temperatureHeatSource)
        self.objHeatSimulator.SetInitalTemp(self.temperatureInitialIsothermal)
        self.objHeatSimulator.UpdatePropertiesTable()
        self.objHeatSimulator.CreateDF()
        return self.objHeatSimulator


class ClassAgent3: 
    temperatureHot = 300
    temperatureCold = 100
    timeHot = 60
    noPtsTemperatureCoarse = 6
    
    def temperatureProfile(self, timeAssess):
        temperatureResult = self.temperatureCold
        if(timeAssess < self.timeHot):
            temperatureResult = self.temperatureHot
        return temperatureResult
    
    def runSimulation(self, timeTotal, timeHot, temperatureHot, temperatureCold, objHeatSimulator):
        score = 0
        
        # run simulation with temperature profile
        
        dTimeRecommend = objHeatSimulator.SuggestedTimeInc(True)
        
        # run the simulator for the hot temperature for a duration of hot time
        timeStart = 0
        timeEnd = timeHot
        noStepsMin =( timeEnd - timeStart)/dTimeRecommend
        noSteps = int(np.ceil(noStepsMin*1.2))
        timeSequence = np.linspace(timeStart, timeEnd, noSteps)
        dTimeLinspace = timeSequence[1] - timeSequence[0]

        objHeatSimulator.TimeInc = dTimeLinspace
        objHeatSimulator.GlobalTime = 0
        objHeatSimulator.SetHeatSource(temperatureHot)
        objHeatSimulator.CreateDF()
        for timeInc in timeSequence[0:-1]:
            objHeatSimulator.CalculateAllNodes() 
        objHeatSimulator.UpdatePropertiesTable()
        
        # run the simulator for the cold temperature until the end of the total time
        timeStart = timeHot
        timeEnd = timeTotal
        noStepsMin =( timeEnd - timeStart)/dTimeRecommend
        noSteps = int(np.ceil(noStepsMin*1.2))
        timeSequence = np.linspace(timeStart, timeEnd, noSteps)
        dTimeLinspace = timeSequence[1] - timeSequence[0]

        objHeatSimulator.TimeInc = dTimeLinspace
        objHeatSimulator.GlobalTime = 0
        objHeatSimulator.SetHeatSource(temperatureCold)
        for timeInc in timeSequence[0:-1]:
            objHeatSimulator.CalculateAllNodes() 
        objHeatSimulator.UpdatePropertiesTable()
        
        self.objHeatSimulator = objHeatSimulator
        
        isTemperatureGoodForCooking = (objHeatSimulator.AllTemp.Node10 > 130) & \
            (objHeatSimulator.AllTemp.Node10 < 135)
        
        scoreOverTime = -np.ones(isTemperatureGoodForCooking.shape)
        scoreOverTime[isTemperatureGoodForCooking] = 1.0
        
        scoreIntegral = np.trapz(scoreOverTime, objHeatSimulator.AllTemp.Time.to_numpy())
        score = scoreIntegral / np.max(objHeatSimulator.AllTemp.Time)

        timeStart = np.min(objHeatSimulator.AllTemp.Time)
        timeEnd = np.max(objHeatSimulator.AllTemp.Time)

        self.timeCoarse = np.linspace(timeStart, timeEnd, self.noPtsTemperatureCoarse)

        self.temperatureCoarse = np.interp(self.timeCoarse, objHeatSimulator.AllTemp.Time, \
            objHeatSimulator.AllTemp.Node10)
        
        return score