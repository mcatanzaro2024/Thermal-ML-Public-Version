import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

class ClassHeatSimulation: 
    #presets
    TempLeftEnd = 200
    Temp_Fluid = 40
    TimeInc = 10
    Density = 7800
    Thermal_Conduct = 50
    Heat_Cap = 470
    Conv_Coeff = 50
    NumNodes = 4
    TotalLength = .1
    Length = 25.0E-3
    Width = 3.0E-3
    Thickness = 3.0E-3
    SurfaceHeat = True
    InitialTemp = 200
    GlobalTime = 0
    ConstThermResistLeft = 0
    ConstThermResistRight = 0
    ConstThermResistCircum = 0
    ThermCapacitance = 0
    Volume = 0
    Temps = [0]
    Tempatures = {'Time':[0]}
    AllTemp = pd.DataFrame(Tempatures)
    SugTime = []
    LocalTemps = []
    Props = []
    PropsTable = pd.DataFrame(Props)
    
    
    def SetNodes(self,InputNumberNodes, InputTotalThickness):
        self.NumNodes = InputNumberNodes
        self.TotalLength = InputTotalThickness
        self.Length = self.TotalLength/self.NumNodes
        #self.SuggestedTimeInc(True)
    def SetSize(self,W,T):
        self.Width = W
        self.Thickness = T
    def SetMaterialProperties(self,InputDensity,InputHeatCapacity,InputThermalConductivity):
        self.Density = InputDensity
        self.Heat_Cap = InputHeatCapacity
        self.Thermal_Conduct = InputThermalConductivity
    def SetConvectionProperties(self,InputConvectionCoeff,InputTempature):
        self.Conv_Coeff = InputConvectionCoeff
        self.Temp_Fluid = InputTempature
    def EnableSurfaceHeatTransfer(self,Boolean):
        self.SurfaceHeat = Boolean
    def SetTimeIncrement(self,InputTimeInc):
        self.TimeInc = InputTimeInc
        #give recomendation based on JP Holman
        #Recommendation = "Confirmed"
        #print(Recommendation)
    def SetHeatSource(self, InputTemp):
        self.TempLeftEnd = InputTemp
    def SetInitalTemp(self,InputInitTemp):
        self.InitialTemp = InputInitTemp
    def SetInitalTempProfile(self,InputArrayTemp):
        #only works if # of nodes Matches
        self.Temps = InputArrayTemp
        #self.CreateDF()
    def SaveDFtoCSV(self, Name):
        #print(self.AllTemp)
        self.AllTemp.to_csv(Name, index=False)
    def AdvanceSimulationByTime(self, timeIncrement, temperatureHeatSource):
        self.SetHeatSource(temperatureHeatSource)
        
        dTimeRecommend = self.SuggestedTimeInc(True)
        
        timeStart = np.max(self.AllTemp.Time)
        timeEnd = timeStart + timeIncrement
        noStepsMin =timeIncrement/dTimeRecommend
        noSteps = int(np.ceil(noStepsMin*1.2))
        timeSequence = np.linspace(timeStart, timeEnd, noSteps)
        dTimeLinspace = timeSequence[1] - timeSequence[0]

        self.TimeInc = dTimeLinspace
        self.GlobalTime = 0
        for timeInc in timeSequence[0:-1]:
            self.CalculateAllNodes() 
        self.UpdatePropertiesTable()
        
    def GetTempatureAtNodeAtTime(self,StartTime,EndTime,Node):
        self.UpdatePropertiesTable()
        if Node > self.NumNodes or Node <= 0:
            return "Error: Node Number Invalid"
        Time = EndTime - StartTime
        if Time % self.TimeInc != 0:
           # print("Time Invalid, rounding...")
            NumSteps = Time/self.TimeInc
            NumSteps = round(NumSteps)
            #print(NumSteps)
            NewTime = NumSteps * self.TimeInc
            #Print = "New Timestep is " + str(NewTime)
           # print(Print)
        else:
            NumSteps = Time/self.TimeInc
        #print(NumSteps)
        if StartTime == 0:
            self.GlobalTime = 0
            self.CreateDF()
            Step = 0
            while Step < NumSteps:
                Step = Step + 1
                self.CalculateAllNodes() 
        elif (abs(StartTime - self.GlobalTime) <= .00001):
            Step = 0
            while Step < NumSteps:
                #print("bruh")
                Step = Step + 1
                self.CalculateAllNodes()
                #print("this works")
        elif (abs(StartTime - self.GlobalTime) > .00001) & (StartTime != 0):
            print("why")
            self.GlobalTime = StartTime
            self.CreateDF()
            if StartTime % self.TimeInc != 0:
                #print("Time Window Invalid, approximating")
                NumPreSteps = StartTime/self.TimeInc
                NumPreSteps = round(NumPreSteps)
            else:
                NumPreSteps = StartTime/self.TimeInc
            PreStep = 0
            while PreStep < NumPreSteps:
                PreStep = PreStep + 1
                self.CalculateAllNodes()
                #print("this works")
            Step = 0
            while Step < NumSteps:
                Step = Step + 1
                self.CalculateAllNodes()
                #print("this works")
        else:
            print("it broke")
        #print(self.GlobalTime)
        self.UpdatePropertiesTable()
        ReturnNode = (Node + 1)
        #print(self.Temps)
        return self.Temps[ReturnNode]
    
    def Properties(self,NodeNumber):
        ConvectionSurfaceArea = self.Width*self.TotalLength
        CrossSectionalArea = self.Width*self.Thickness
        self.Volume = self.Width*self.Thickness*self.Length
        self.ConstThermResistLeft = self.Length/(CrossSectionalArea*self.Thermal_Conduct)
        if NodeNumber == self.NumNodes:
            # for final Node
            self.ConstThermResistRight = 0
            self.ConstThermResistCircum = 2/(ConvectionSurfaceArea*self.Conv_Coeff)
            self.ThermCapacitance = self.Density*self.Volume*self.Heat_Cap/2
        else:  
            self.ConstThermResistRight = self.Length/(CrossSectionalArea*self.Thermal_Conduct)
            self.ConstThermResistCircum = 1/(ConvectionSurfaceArea*self.Conv_Coeff)
            self.ThermCapacitance = self.Density*self.Volume*self.Heat_Cap
        return 
    def CalcTemp(self,LeftTemp, SelfTemp, RightTemp,NodeN):
        self.Properties(NodeN)
        ResistL = (LeftTemp - SelfTemp)/self.ConstThermResistLeft
        if NodeN == self.NumNodes:
            # for final Node
            ResistR = 0
        else:
            ResistR = (RightTemp - SelfTemp)/self.ConstThermResistRight
        
        if self.SurfaceHeat == True:
            ResistC = (self.Temp_Fluid - SelfTemp)/self.ConstThermResistCircum
        else:
            ResistC = 0
        Temperature = self.TimeInc/self.ThermCapacitance*(ResistL + ResistR + ResistC)+SelfTemp
        return Temperature 
    def UpdatePropertiesTable(self):
        self.Props = {'Node':[1]}
        self.PropsTable = pd.DataFrame(self.Props)
        Values = [0,0,0,0,0,0]
        a = 0
        self.PropsTable['R_Left'] = 0
        self.PropsTable['R_Right'] = 0
        self.PropsTable['R_Top'] = 0
        self.PropsTable['ThermCap'] = 0
        self.PropsTable['C/(1/R)'] = 0
        while a < self.NumNodes:
            a = a + 1
            self.Properties(a)
            Values[0] = a
            Values[1] = self.ConstThermResistLeft
            Values[2] = self.ConstThermResistRight
            if self.SurfaceHeat == False:
                Values[3] = 0
            elif self.SurfaceHeat == True:
                Values[3] = self.ConstThermResistCircum
            Values[4] = self.ThermCapacitance
            if self.SurfaceHeat == False:
                if a == self.NumNodes:
                    #print(Values)
                    Values[5] = (Values[4])/(1/Values[1])
                else:
                    Values[5] = (Values[4])/((1/Values[1])+(1/Values[2]))
            else:
                if a == self.NumNodes:
                    Values[5] = (Values[4])/((1/Values[1])+(1/Values[3]))
                else:
                    Values[5] = (Values[4])/((1/Values[1])+(1/Values[2])+(1/Values[3]))
            if a == 1:
                self.PropsTable.loc[0] = Values
            else:
                self.PropsTable.loc[len(self.PropsTable.index)] = Values
        #print(self.PropsTable)
        
        
        
        
        
        
    def CreateDF(self):
        self.Tempatures = {'Time':[0],'Heater':[self.TempLeftEnd]}
        self.AllTemp = pd.DataFrame(self.Tempatures)
        self.InputTemp(self.InitialTemp)
        a = 0
        while a < self.NumNodes:
            a = a + 1
            ColumnName = 'Node' + str(a)
            self.AllTemp[ColumnName] = [self.Temps[a + 1]]
        self.AllTemp['Convection'] = self.Temp_Fluid
    def UpdateDF(self, CurrentTemp):
        self.AllTemp.loc[len(self.AllTemp.index)] = CurrentTemp
        #print(self.AllTemp)
    def CalculateAllNodes(self):
        self.LocalTemps.clear()
        self.LocalTemps.append(0)
        #print(self.Temps)
        LNode = 1
        while LNode <= self.NumNodes:
            LNode = LNode + 1
            BeforeLocalNode = LNode - 1
            
            if BeforeLocalNode <= 1:
                BeforeLocalTemp = self.TempLeftEnd
            else:
               #
                BeforeLocalTemp = self.Temps[BeforeLocalNode]

            #print(self.Temps)
            LocalTemp = self.Temps[LNode]
            AfterLocalNode = LNode + 1
            
            if AfterLocalNode > (self.NumNodes + 1):
                AfterLocalTemp = self.Temp_Fluid
            else:
                AfterLocalTemp = self.Temps[AfterLocalNode]
            #print(BeforeLocalTemp, LocalTemp, AfterLocalTemp, (LNode-1))
            LocalTemp = self.CalcTemp(BeforeLocalTemp, LocalTemp, AfterLocalTemp, (LNode-1))
            self.LocalTemps.append(LocalTemp)
        L = 0
        self.Temps[1] = self.TempLeftEnd
        while L < self.NumNodes:
            L = L + 1
            #print(L)
            #print(self.Temps)
            #print(self.LocalTemps)
            self.Temps[L + 1] = self.LocalTemps[L]
        self.Temps[L+2] = self.Temp_Fluid
      
        self.GlobalTime = self.Temps[0] + self.TimeInc
        #print(self.GlobalTime)
        self.Temps[0] = self.GlobalTime
        self.UpdateDF(self.Temps)
        #print(self.Temps)
    def InputTemp(self,StartTemp):
        a = 0
        self.Temps = [0]
        self.Temps.append(self.TempLeftEnd)
        while a < self.NumNodes:
            a = a + 1
            self.Temps.append(StartTemp)
        self.Temps.append(self.Temp_Fluid)
        #print(self.Temps)
    #time step recomendation
    def SuggestedTimeInc(self,Return):
        NN= 0
        self.SugTime.clear()
        self.UpdatePropertiesTable()
        while NN < 4:
            NN = NN + 1
            self.Properties(NN)
            TimeSuggestion = (self.PropsTable.iloc[(NN-1),5])
            #print(TimeSuggestion)
            TimeSuggestion = TimeSuggestion - (.4)*(TimeSuggestion)
            if (round(TimeSuggestion)) > 2:
                self.SugTime.append(round(TimeSuggestion))
            elif (round(TimeSuggestion,1)) > .2:
                self.SugTime.append(round(TimeSuggestion,1))
            elif (round(TimeSuggestion,2)) > .02:
                self.SugTime.append(round(TimeSuggestion,2))
            elif (round(TimeSuggestion,3)) > .002:
                self.SugTime.append(round(TimeSuggestion,3))
            elif (round(TimeSuggestion,4)) > .0002:
                self.SugTime.append(round(TimeSuggestion,4))
            else:
                self.SugTime.append(round(TimeSuggestion,5))
        SugTimeInc = min(self.SugTime)
        if Return == True:
            #print(SugTimeInc)
            return SugTimeInc
    def GraphResults(self,Title):
        Trial = self.AllTemp
        plt.figure()
        plt.plot(Trial.Time, Trial.Heater, ":k", label="HeatSource")
        plt.plot(Trial.Time, Trial.Convection, "--k", label="Convection")
        Node = 0
        while Node < self.NumNodes:
            Node = Node + 1
            TrialValue = Node + 1
            LabelValue = "Node" + str(Node)
            plt.plot(Trial.Time, Trial.iloc[:,TrialValue], label=LabelValue)
        plt.xlabel("Time")
        plt.ylabel("Temperature (C)")
        plt.title(Title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.show()