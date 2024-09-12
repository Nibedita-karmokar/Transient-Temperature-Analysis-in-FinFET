#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:54:00 2024

@author: karmo005
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:18:15 2024

@author: karmo005
"""



import numpy as np
from numpy import *
import json
import matplotlib.pyplot as plt
from matplotlib import pyplot as patches
import time


from scipy.interpolate import griddata
import Transient_FDM

def multiplier(lst, constant):
    result=[]
    for num in lst:
        result.append(num * constant)
    return result

#NXP
class transistor(object):

    width=166
    Abeta=0.003
    #Avt=1 #mVum
    Avt=1*10**(-6)      #Vnm
    gm_over_I=2.5
    t_max=11*10**(-9)
    time_step=0.5*10**(-9)
    t_on=5*10**(-9)
    t_off=5*10**(-9)  
    t_period=10*10**(-9) 
    I_u=51*10**(-6) 
    voltage=1.8 #V
    
    sx=90
    sy=90 

    line_w=1
    gate_width=1
    drain_width=1
    source_width=1
    scale_factor=1
    Fin_number=8
    Fin_number_power=8
    Fin_pitch=45
    height=Fin_pitch*(Fin_number-1)
    height_12=Fin_pitch*(Fin_number_power-1)
    Gate_number=1
    thermal_conductivity={}
    density={}
    heat_capacity={}
    
    current=I_u*Fin_number #A
    current_power=I_u*Fin_number_power

    power_d=0.1
    
    poly_to_wire=17
    poly_width=14
    via_number_S=0 

    t_sub2gnd=960
    t_cnt2gnd=3500
    t_gox=1.8
    t_gate=44
    t_chnl=32
    t_box=140

    t_substrate=60
    depth=t_substrate+t_box+t_gate

    Metal_via_depth=0
    M_via_depth=0    
    zk_fin_top=0
    zk_fin_bottom=0
    rho_metal=1.72*10**(-8)  
    poly_gap=76
    T_0=273
    mu_eff_0=100*10**(14) 
    lambda_v=-2*10^(-3) #V/K 
    V_T0=0.2
    V_GS=0.56
    beta=100
    Af_1=0.0085
    d0=1
    rho_u=0.9
    #transistor.Af_1, transistor.current, list1, transistor.rho_u, transistor.d0

    def __init__(self, x, y, transistor_name):
        self.x=x
        self.y=y
        self.transistor_name=transistor_name
        self.transistor_id={}
        self.S_mark=0
        self.D_mark=0
        
        self.via_number_D=0
        self.inter_connected=[]
        self.x_coord=0
        self.y_coord=0

class transistor_AD(object):

    def __init__(self):
        self.Fin_regions=[]
        self.Poly_regions=[]   
        self.Routing_channel_regions=[]
        self.Dummy_regions = []
        self.Substrate_regions=[]
        self.FinFET_regions=[]
        self.Routing_region=[]
        self.Power_region=[]
        self.V0_regions=[]
        self.V1_regions=[]
        self.M1_regions=[]
        self.M2_regions=[]
        self.M3_regions=[] 

class FinFET(object):   
    
    def __init__(self, x, y, name):
        self.x=x
        self.y=y
        self.name=name
        self.identity={}
        self.S_visit=0
        self.D_visit=0
        
        self.mark=0
        self.global_id={}
        
        self.via_number_D=0
        self.connected=[]
        self.inter_connect_length=0
        self.routed=0
        self.subgroup_hourizontal_routed=0
        self.connected_wire={}
        self.track_left_global=[]
        self.track_right_global=[]
        
        self.all_htracks=[]



class Poly:
    scale_factor=1
    def __init__(self, layer, netName, direction):
        self.layer=layer
        self.netName=netName
        self.direction=direction
        
    def segment(self, mat, r, y_unequal_list, j, x0, y0, z0, x1, y1, z1, netType, Poly_regions, Dummy_regions):
        cube=[x0*Poly.scale_factor, y0*Poly.scale_factor, z0*Poly.scale_factor,  x1*Poly.scale_factor, y1*Poly.scale_factor, z1*Poly.scale_factor]
        Poly_regions.append(cube)
        rect=[x0*Poly.scale_factor, y0*Poly.scale_factor,  x1*Poly.scale_factor, y1*Poly.scale_factor]

        for i in range (1, r-1):
            if mat[i][j]==2:
                #print('mat values', j, mat[1][j])
                y1d=y_unequal_list[i]-transistor.height/2
                y2d=y_unequal_list[i]+transistor.height/2
                cube_dummy=[x0*Poly.scale_factor, y1d*Poly.scale_factor, z0*Poly.scale_factor,  x1*Poly.scale_factor, y2d*Poly.scale_factor, z1*Poly.scale_factor]
                Dummy_regions.append(cube_dummy)

        
        terminal = {'layer' : self.layer, 'netName' : self.netName, 'rect' : rect}
        if netType in ['drawing', 'pin', 'blockage']:
            terminal['netType']=netType
        else:
            assert "Invalid net type, valid net types include drawing, pin, and blockage"               
        
        return terminal, Poly_regions, Dummy_regions
    
class Metal:
    scale_factor=1
    def __init__(self, layer, netName, direction):
        self.layer=layer
        self.netName=netName
        self.direction=direction
        
    
    def segment(self, x0, y0, z0, x1, y1, z1, netType, M1_regions):
        cube=[x0*Metal.scale_factor, y0*Metal.scale_factor, z0*Metal.scale_factor,  x1*Metal.scale_factor, y1*Metal.scale_factor, z1*Metal.scale_factor]
        M1_regions.append(cube)
        rect=[x0*Metal.scale_factor, y0*Metal.scale_factor,  x1*Metal.scale_factor, y1*Metal.scale_factor]
        
        terminal = {'layer' : self.layer, 'netName' : self.netName, 'rect' : rect}
        if netType in ['drawing', 'pin', 'blockage']:
            terminal['netType']=netType
        else:
            assert "Invalid net type, valid net types include drawing, pin, and blockage"               
        
        return terminal, M1_regions
    
class Fin:
    scale_factor=1
    def __init__(self, layer, netName, direction):
        self.layer=layer
        self.netName=netName
        self.direction=direction
        
    
    def segment(self, x0, y0, z0, x1, y1, z1, netType, Fin_regions):
        cube=[x0*Fin.scale_factor, y0*Fin.scale_factor, z0*Fin.scale_factor, x1*Fin.scale_factor, y1*Fin.scale_factor, z1*Fin.scale_factor]
        Fin_regions.append(cube)
        rect=[x0*Fin.scale_factor, y0*Fin.scale_factor, x1*Fin.scale_factor, y1*Fin.scale_factor]
        terminal = {'layer' : self.layer, 'netName' : self.netName, 'rect' : rect}
        if netType in ['drawing', 'pin', 'blockage']:
            terminal['netType']=netType
        else:
            assert "Invalid net type, valid net types include drawing, pin, and blockage"               
        
        return terminal, Fin_regions


def Metal_layer_information(k):
    M1={}
    M1['Direction']=k['Direction']
    M1['Pitch']=k['Pitch']/transistor.scale_factor
    M1['Space']=(k['Pitch']-k['Width'])/transistor.scale_factor
    M1['Width']=k['Width']/transistor.scale_factor
    M1['Color']=k['Color'] 
    return M1

def Via_layer_information(k):
    V1={}
    V1['WidthX']=k['WidthX']/transistor.scale_factor
    V1['WidthY']=k['WidthY']/transistor.scale_factor
    V1['VencA_L']=k['VencA_L']/transistor.scale_factor
    V1['VencA_H']=k['VencA_H']/transistor.scale_factor
    V1['VencP_L']=k['VencP_L']/transistor.scale_factor
    V1['VencP_H']=k['VencP_H']/transistor.scale_factor     
    return V1

def Poly_information(k):
    Poly={}
    Poly['Direction']=k['Direction']
    Poly['Pitch']=k['Pitch']/transistor.scale_factor
    Poly['Space']=(k['Pitch']-k['Width'])/transistor.scale_factor
    Poly['Width']=k['Width']/transistor.scale_factor
    return Poly

def Fin_information(k):
    Fin={}
    Fin['Direction']=k['Direction']
    Fin['Pitch']=k['Pitch']/transistor.scale_factor
    Fin['Space']=(k['Pitch']-k['Width'])/transistor.scale_factor
    Fin['Width']=k['Width']/transistor.scale_factor
    return Fin


def CC_plot(layout_width, layout_height, Device_Name, mat, row, column, transistor, line_width, Layers_dict, obj_tran):
    layout_aspect_ratio = layout_width / layout_height
    fig_width=6.0    
    # Calculate the height of the figure to match the layout aspect ratio
    fig_height = fig_width / layout_aspect_ratio
    
    # Create a Matplotlib figure with the calculated size
    plt.figure(figsize=(fig_width, fig_height))
    for D in Device_Name:    
        for i in range (row):    
            for j in range (column):
                if mat[i][j]==D and D!=2:
                    xk=obj_tran[i][j].x_coord
                    yk=obj_tran[i][j].y_coord


                    color_num='k'
                    var=str(int(mat[i][j]))    
                    plt.text(xk-(transistor.sx/5), yk-(transistor.sx/3), var, fontsize=8, color=color_num, fontweight='bold')
                    
                    x1=xk-transistor.width/2
                    y1=yk+transistor.height/2
                    x2=xk+transistor.width/2    #top plate
                    y2=yk+transistor.height/2                    
                    plt.plot([x1,x2],[y1,y2], color='r', linewidth=line_width) 

                    
    plt.tick_params(axis='both', labelsize=14)                    

def Gate_port(mat, r, y_unequal_list, j, Layers_dict, Terminal_list, AD_obj, transistor, x, y, z0, z1):
    multiplier = 1
    if y_unequal_list[len(y_unequal_list)-1]==-y_unequal_list[0]:
        multiplier = 2
    plt.gca().add_patch(patches.Rectangle((x-Layers_dict['Poly']['Width']/2, y_unequal_list[len(y_unequal_list)-1]-transistor.height_12/2-Layers_dict['Fin']['Width']/2), Layers_dict['Poly']['Width'], multiplier*(y+transistor.height_12/2+Layers_dict['Fin']['Width']/2), color='magenta'))
    wire_segment_1=Poly('Poly', 'null', Layers_dict['Poly']['Direction'])
    (terminal, AD_obj.Poly_regions, AD_obj.Dummy_regions)=wire_segment_1.segment(mat, r, y_unequal_list, j, x-Layers_dict['Poly']['Width']/2, y_unequal_list[len(y_unequal_list)-1]-transistor.height_12/2-Layers_dict['Fin']['Width']/2, z0, x+Layers_dict['Poly']['Width']/2, y+transistor.height_12/2+Layers_dict['Fin']['Width']/2, z1, 'drawing', AD_obj.Poly_regions, AD_obj.Dummy_regions)
    Terminal_list.append(terminal)

def S_D_port(Layers_dict, Terminal_list, AD_obj, transistor, x, y, z0, z1):
    plt.gca().add_patch(patches.Rectangle((x-Layers_dict['M1']['Width']/2, y-transistor.height/2-Layers_dict['Fin']['Width']/2), Layers_dict['M1']['Width'], transistor.height+Layers_dict['Fin']['Width'], color='cyan'))
    wire_segment_1=Metal('M1', 'null', Layers_dict['M1']['Direction'])
    (terminal, AD_obj.M1_regions)=wire_segment_1.segment(x-Layers_dict['M1']['Width']/2, y-transistor.height/2, z0, x+Layers_dict['M1']['Width']/2, y+transistor.height/2, z1, 'drawing', AD_obj.M1_regions)
    Terminal_list.append(terminal)
    
def Fin_place(Layers_dict, Terminal_list, AD_obj, transistor, x, y, zk_fin_top, zk_fin_bottom, final):
    if final==0:
        y=y+transistor.height/2-Layers_dict['Fin']['Width']/2
        for i in range (transistor.Fin_number):   
                
            plt.gca().add_patch(patches.Rectangle((x-transistor.width/2-Layers_dict['Poly']['Width']/2, y), transistor.width+Layers_dict['Poly']['Width'], Layers_dict['Fin']['Width'], color='b'))
            wire_segment_1=Fin('Fin', 'null', Layers_dict['Fin']['Direction'])
    
            (terminal, AD_obj.Fin_regions)=wire_segment_1.segment(x-transistor.width/2-Layers_dict['Poly']['Width']/2, y, zk_fin_bottom, x+transistor.width/2+Layers_dict['Poly']['Width']/2, y+Layers_dict['Fin']['Width'], zk_fin_top, 'drawing', AD_obj.Fin_regions)
            Terminal_list.append(terminal)
    
            if i!=transistor.Fin_number-1:
                cube=[x-transistor.width/2-Layers_dict['Poly']['Width']/2, y-Layers_dict['Fin']['Pitch']+Layers_dict['Fin']['Width'] , zk_fin_bottom, x+transistor.width/2+Layers_dict['Poly']['Width']/2, y, zk_fin_top]
                AD_obj.Substrate_regions.append(cube)
            y=y-(Layers_dict['Fin']['Pitch'])

    else:
        y=y+transistor.height_12/2-Layers_dict['Fin']['Width']/2
        for i in range (transistor.Fin_number_power):    

            plt.gca().add_patch(patches.Rectangle((x-transistor.width/2-Layers_dict['Poly']['Width']/2, y), transistor.width+Layers_dict['Poly']['Width'], Layers_dict['Fin']['Width'], color='b'))
            wire_segment_1=Fin('Fin', 'null', Layers_dict['Fin']['Direction'])
    
            (terminal, AD_obj.Fin_regions)=wire_segment_1.segment(x-transistor.width/2-Layers_dict['Poly']['Width']/2, y, zk_fin_bottom, x+transistor.width/2+Layers_dict['Poly']['Width']/2, y+Layers_dict['Fin']['Width'], zk_fin_top, 'drawing', AD_obj.Fin_regions)
            Terminal_list.append(terminal)
    
            if i!=transistor.Fin_number_power-1:
                cube=[x-transistor.width/2-Layers_dict['Poly']['Width']/2, y-Layers_dict['Fin']['Pitch']+Layers_dict['Fin']['Width'] , zk_fin_bottom, x+transistor.width/2+Layers_dict['Poly']['Width']/2, y, zk_fin_top]
                AD_obj.Substrate_regions.append(cube)

            y=y-Layers_dict['Fin']['Pitch']

def global_tracks(column, obj_FinFET):
    max_track=10
    
    first_init=0
    init=max_track
    for col in range (column):                
        for i in range (len(obj_FinFET)):
            for j in range (len(obj_FinFET[i])):
                if obj_FinFET[i][j].x==col:
                    for k in range (first_init, init):                   
                        obj_FinFET[i][j].track_left_global.append(k)
                    for k in range (init, init+max_track):                       
                        obj_FinFET[i][j].track_right_global.append(k)
            
        first_init= init              
        init=init+max_track  

def Temperature_plot(x, y, Temperature, fig_name):
    plt.figure(figsize=(7, 8))
    xmin = min(x)
    xmax=max(x)
    ymin = min(y)
    ymax=max(y)
   
    grid_x, grid_y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
    zi = griddata((x,y), Temperature, (grid_x, grid_y), method='linear')
    plt.figure()

    plt.imshow(zi[:,:,0].T,cmap='jet',extent=[xmin,xmax,ymin,ymax], origin="lower")
    cb=plt.colorbar(shrink=0.75)
    cb.set_label('Temperature rise (K)')    

    plt.savefig(fig_name, dpi=300)  

def Power_plot(x, y, Power, fig_name):
    plt.figure(figsize=(7, 8))
    xmin = min(x)
    xmax=max(x)
    ymin = min(y)
    ymax=max(y)
   
    grid_x, grid_y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
    zi = griddata((x,y), Power, (grid_x, grid_y), method='linear')
    plt.figure()

    plt.imshow(zi[:,:,0].T,cmap='jet',extent=[xmin,xmax,ymin,ymax], origin="lower")
    cb=plt.colorbar(shrink=0.75)
    cb.set_label('Power(W)')    

    plt.savefig(fig_name, dpi=300)       



def find_divisors(number):
    divisors = []
    
    for i in range(1, number + 1):
        if number % i == 0:
            divisors.append(i)
    
    return divisors

# Example usage

def find_combinations(divisors, target):
    combinations = []
    
    for i in range(len(divisors)):
        for j in range(i, len(divisors)):
            num1 = divisors[i]
            num2 = divisors[j]
            
            if num1 * num2 == target:
                combinations.append((num1, num2))
    
    return combinations

def main(json_file, finger_count, duty_cycle, current_per_fin): 

    Layers_dict={}
    
    with open(json_file, "rt") as fp:
        j = json.load(fp)
        sf=1
    
        transistor.thermal_conductivity['gate']=j['thermal_conductivity']['gate']*sf
        transistor.thermal_conductivity['SiO2']=j['thermal_conductivity']['SiO2']*sf
        transistor.thermal_conductivity['Si substrate']=j['thermal_conductivity']['Si substrate']*sf
        transistor.thermal_conductivity['Si NMOS fin']=j['thermal_conductivity']['Si NMOS fin']*sf
        transistor.thermal_conductivity['Si NMOS SD']=j['thermal_conductivity']['Si NMOS SD']*sf
        transistor.thermal_conductivity['contact']=j['thermal_conductivity']['contact']*sf
    
        transistor.heat_capacity['gate']=j['heat_capacity']['gate']
        transistor.heat_capacity['Si']=j['heat_capacity']['Si']
        transistor.heat_capacity['SiO2']=j['heat_capacity']['SiO2']
        transistor.heat_capacity['contact']=j['heat_capacity']['contact']
        
        transistor.density['gate']=j['density']['gate']
        transistor.density['Si']=j['density']['Si']
        transistor.density['SiO2']=j['density']['SiO2']
        transistor.density['contact']=j['density']['contact']
    
        for i in range (len(j['Abstraction'])):
            if j['Abstraction'][i]['Layer']=='Poly': 
                Layers_dict['Poly']=Poly_information(j['Abstraction'][i])
            if j['Abstraction'][i]['Layer']=='Fin': 
                Layers_dict['Fin']=Fin_information(j['Abstraction'][i])            
            if j['Abstraction'][i]['Layer']=='M1': 
                Layers_dict['M1']=Metal_layer_information(j['Abstraction'][i])
            if j['Abstraction'][i]['Layer']=='V0': 
                Layers_dict['V0']=Via_layer_information(j['Abstraction'][i])            
            if j['Abstraction'][i]['Layer']=='V1': 
                Layers_dict['V1']=Via_layer_information(j['Abstraction'][i])
            if j['Abstraction'][i]['Layer']=='M2': 
                Layers_dict['M2']=Metal_layer_information(j['Abstraction'][i])
            if j['Abstraction'][i]['Layer']=='V2': 
                Layers_dict['V2']=Via_layer_information(j['Abstraction'][i])
            if j['Abstraction'][i]['Layer']=='M3': 
                Layers_dict['M3']=Metal_layer_information(j['Abstraction'][i]) 
        

    transistor.width=2*Layers_dict['M1']['Pitch']-Layers_dict['Poly']['Width']
    transistor.height=Layers_dict['Fin']['Pitch']*(transistor.Fin_number-1)
    transistor.height_12=Layers_dict['Fin']['Pitch']*(transistor.Fin_number_power-1)
    transistor.sx=Layers_dict['M1']['Pitch']
    transistor.sy=Layers_dict['M2']['Pitch']
    transistor.poly_width=Layers_dict['Poly']['Width']
    transistor.poly_gap=Layers_dict['Poly']['Pitch']-Layers_dict['Poly']['Width']
    transistor.poly_to_wire=(transistor.poly_gap-Layers_dict['M1']['Width'])/2
    
    Finger_count = input ("Enter Finger Count: ")
    Diffusion_type='Share'
    list1=[int(Finger_count)]
    
    Active_Dummy = [[1, 0]]
    
    Device_Name=[]
    
    for i in range (len(list1)):
        Device_Name.append(i+1)
    
    print('Device_Name', Device_Name)
    
    number=int(Finger_count)
    divisors_initial = find_divisors(number)
    combinations_initial = find_combinations(divisors_initial, number)
        
    len_comb_initial=len(combinations_initial)
    r_initial_2=combinations_initial[len_comb_initial-1][0]
    c_initial_2=combinations_initial[len_comb_initial-1][1]
    
    r_initial_0=combinations_initial[len_comb_initial-2][0]
    c_initial_0=combinations_initial[len_comb_initial-2][1]
    
    finger_count_list=[finger_count]   #C
    
    percentage =[duty_cycle]
    I_u=[current_per_fin]
    
    Voltages=[1]
    
    start_time = time.time()
    
    AD=0
    for duty in range (len(percentage)):
        transistor.t_on=transistor.t_period*percentage[duty]
        transistor.t_off=transistor.t_period-transistor.t_on
        
        for nf in range (len(finger_count_list)):    
            number=finger_count_list[nf]
            transistor.current=I_u[nf]*transistor.Fin_number #A
            transistor.voltage=Voltages[nf]
            
            divisors_initial = find_divisors(number)
            combinations_initial = find_combinations(divisors_initial, number)
                
            len_comb_initial=len(combinations_initial)
            
            r_initial_2=combinations_initial[len_comb_initial-2][0]
            c_initial_2=combinations_initial[len_comb_initial-2][1] 
    
            if AD>=2:
                r_initial_2=combinations_initial[len_comb_initial-1][0]
                c_initial_2=combinations_initial[len_comb_initial-1][1]  
            
            if finger_count_list[nf]==32:
                r_initial_2=combinations_initial[len_comb_initial-1][0]
                c_initial_2=combinations_initial[len_comb_initial-1][1]
    
            
            r_initial_0=combinations_initial[len_comb_initial-2][0]
            c_initial_0=combinations_initial[len_comb_initial-2][1]
        
            list1=[]
        
            
            Active_regions=c_initial_2/Active_Dummy[AD][0]
            Dummy_regions=0
            if Active_Dummy[AD][1]!=0:        
                Dummy_regions=Active_regions+1
            
            Dummy_column=Active_Dummy[AD][1]*Dummy_regions
            
            
            Active=int(finger_count_list[nf])
            Dummy=int(Dummy_column*r_initial_2)
            
            if Dummy!=0:
                list1=[Active, Dummy]
            if Dummy==0:
                list1=[Active]        
            
            Device_Name=[]
            
            for i in range (len(list1)):
                Device_Name.append(i+1)
            
        
            Active_column=Active_regions*Active_Dummy[AD][0]
            
            r_2=r_initial_2
            c_2=int(Active_column+Dummy_column)
            
            if Active_Dummy[AD][1]==0: 
                r_2=r_initial_0
                c_2=c_initial_0      
            
            
            row=r_2
            column=c_2
            
            len_list=len(list1)
            
            mat=[]
            S_D_mat=[]
            placement_type='Spiral'
               
            mat_1 = []
            
            if Active_Dummy[AD][1]==0:       
                mat_1=np.array([[1] * column for _ in range(row)])
        
            else:
                for i in range (row):
                    k=[]
                    count_c=0
                    while count_c<column:
                        for Dum in range (Active_Dummy[AD][1]):
                            count_c=count_c+1
                            k.append(2)
                        if count_c!=column:
                            for Act in range (Active_Dummy[AD][0]):
                                count_c=count_c+1
                                k.append(1)
                                
                    mat_1.append(k)
                    
                mat_1=np.array(mat_1)
        
            r_1=len(mat_1)
            c_1=len(mat_1[0])
            
            orig_shape = mat_1.shape
            
            # calculate the new shape of the padded matrix
            new_shape = (orig_shape[0] + 2, orig_shape[1] + 2)
    
            
            r=new_shape[0]
            c=new_shape[1]
    
            mat = np.zeros(new_shape, dtype=mat_1.dtype)
            mat[1:-1, 1:-1] = mat_1
            
                    
            print(' placement matrix', mat)
            S_D_mat_1 =[['' for j in range(c_1+1)] for i in range(r_1)]
    
            
            i_1=0
            for i in range (r_1):
                start='D'
                for j in range (c_1-1):     
                    if mat_1[i][j]==mat_1[i][j+1]:
                        start='S'
                        break
                
                if start=='S':        
                    for j in range (0, c_1, 2):  
                        S_D_mat_1[i][j]='S'
                        S_D_mat_1[i][j+1]='D'
                    S_D_mat_1[i][c_1]='S'
                        
                if start=='D':        
                    for j in range (0, c_1, 2): 
                        S_D_mat_1[i][j]='D'
                        S_D_mat_1[i][j+1]='S'
                    S_D_mat_1[i][c_1]='D'            
    
            new_shape = (r, c+1)        
            S_D_mat = [['' for j in range(new_shape[1])] for i in range(new_shape[0])]               
            
            for i in range (r_1):
                for j in range (c_1+1):
                    S_D_mat[i+1][j+1]=S_D_mat_1[i][j]
                
                    
            for i in range (r_1):
                if S_D_mat_1[i][0]=='S':   
                    S_D_mat[i+1][0]='D'
                    S_D_mat[i+1][new_shape[1]-1]='D'
                elif S_D_mat_1[i][0]=='D':   
                    S_D_mat[i+1][0]='S'
                    S_D_mat[i+1][new_shape[1]-1]='S'
                    
            for k in range (new_shape[0]-r_1-1, -1, -1):      
                for i in range (new_shape[1]):
                    S_D_mat[k][i]=S_D_mat[k+1][i]
                    S_D_mat[(new_shape[0])-(k+1)][i]=S_D_mat[new_shape[0]-(k+2)][i]
    
            Diffusion=[[0]*c]*r
            
            if Diffusion_type=='Share':
                Diffusion=[['Share']*c]*r
            
            Terminal_list=[]
            
            obj_tran=[]
            for i in range (r_1):
                q=[]
                for j in range (c_1):
                    q.append(transistor(j, i, mat_1[i][j]))
                obj_tran.append(q)
            
            fin_track=2
            track_num=[]
            for i in range (r-1):
                track_num.append(fin_track)
                    
            x_unequal_list=[]
            if c%2==0:    
                x_val=0
                if Diffusion[0][int(c/2)-1]=='Break':
                    x_val=transistor.sx/2+transistor.width/2+Layers_dict['Poly']['Width']/2
                elif Diffusion[0][int(c/2)-1]=='Share':
                    x_val=transistor.poly_gap/2+Layers_dict['Poly']['Width']/2
                x_unequal_list.append(x_val)
                x_unequal_list.append(-x_val)
                for i in range (c//2, c-1):
                    if Diffusion[0][i]=='Break':
                        x_val=x_val+transistor.sx+transistor.width+Layers_dict['Poly']['Width']
                    elif Diffusion[0][i]=='Share':
                        x_val=x_val+transistor.poly_gap+Layers_dict['Poly']['Width']
            
                    x_unequal_list.append(x_val)
                    x_unequal_list.append(-x_val)
            else:    
                x_val=0
                x_unequal_list.append(x_val)
                if Diffusion[0][int(c/2)-1]=='Break':
                    x_val=transistor.sx+transistor.width+Layers_dict['Poly']['Width']
                elif Diffusion[0][int(c/2)-1]=='Share':
                    x_val=transistor.poly_gap+Layers_dict['Poly']['Width']
                x_unequal_list.append(x_val)
                x_unequal_list.append(-x_val)
                for i in range (c//2+1, c-1):
                    if Diffusion[0][i]=='Break':
                        x_val=x_val+transistor.sx+transistor.width+Layers_dict['Poly']['Width']
                    elif Diffusion[0][i]=='Share':
                        x_val=x_val+transistor.poly_gap+Layers_dict['Poly']['Width']
            
                    x_unequal_list.append(x_val)
                    x_unequal_list.append(-x_val)
            
            
            min_x=min(x_unequal_list)
            x_new=0
            for i in range (len(x_unequal_list)):
                x_new=x_unequal_list[i]-min_x+transistor.width/2+Layers_dict['Poly']['Width']/2
                x_unequal_list[i]=x_new
                
    
            x_unequal_list.sort()
            
            y_unequal_list=[]
            if r%2==0:
                y_val=track_num[int(r/2)-1]/2*Layers_dict['Fin']['Pitch']+transistor.height/2
                y_unequal_list.append(y_val)
                y_unequal_list.append(-y_val)
                for i in range (r//2, r-2):
                    y_val = y_val   + track_num[i]*Layers_dict['Fin']['Pitch']+transistor.height
                    y_unequal_list.append(y_val)
                    y_unequal_list.append(-y_val)
                    
                for i in range (r-2, r-1):
                    y_val = y_val   + track_num[i]*Layers_dict['Fin']['Pitch']+transistor.height/2+transistor.height_12/2
                    y_unequal_list.append(y_val)
                    y_unequal_list.append(-y_val)
            else:
                y_val=0
                y_unequal_list.append(0)
                for i in range (r//2, r-2):
                    y_val = y_val   + track_num[i]*Layers_dict['Fin']['Pitch']+transistor.height
                    y_unequal_list.append(y_val)
                    y_unequal_list.append(-y_val)
                    
                for i in range (r-2, r-1):
                    y_val = y_val   + track_num[i]*Layers_dict['Fin']['Pitch']+transistor.height/2+transistor.height_12/2
                    y_unequal_list.append(y_val)
                    y_unequal_list.append(-y_val)
            
            min_y=min(y_unequal_list)
            y_new=0
            for i in range (len(y_unequal_list)):
                y_new=y_unequal_list[i]-min_y+transistor.height/2+Layers_dict['Fin']['Width']/2
                y_unequal_list[i]=y_new
            
            y_unequal_list.sort(reverse=True)
            x_list_r=[]
            y_list_r=[]
            
            for i in range ((r-r_1)//2, len(y_unequal_list)-(r-r_1)//2):
                y_list_r.append(y_unequal_list[i])
                
        
            for i in range ((c-c_1)//2, len(x_unequal_list)-(c-c_1)//2):
                x_list_r.append(x_unequal_list[i])
            
                
            for i in range (len(obj_tran)):
                for j in range (len(obj_tran[i])):
                    obj_tran[i][j].x_coord=x_list_r[j]
                    obj_tran[i][j].y_coord=y_list_r[i]
            
            AD_obj=transistor_AD()
            
            xk=x_unequal_list[len(x_unequal_list)-1]
            
            transistor.zk_fin_top=(transistor.depth+transistor.Metal_via_depth)/2-transistor.Metal_via_depth-transistor.t_gox-(transistor.t_gate-transistor.t_chnl-transistor.t_gox)
            transistor.zk_fin_bottom=(transistor.depth+transistor.Metal_via_depth)/2-transistor.Metal_via_depth-transistor.t_gate-transistor.t_box
            
            zk_fin_top=(transistor.depth+transistor.Metal_via_depth)/2-transistor.Metal_via_depth-transistor.t_gox-(transistor.t_gate-transistor.t_chnl-transistor.t_gox)
            zk_fin_bottom=(transistor.depth+transistor.Metal_via_depth)/2-transistor.Metal_via_depth-transistor.t_gate-transistor.t_box
            

            for j in range (c-1, -1, -1):
                xk=x_unequal_list[j]
                for i in range (r):
                    yk=y_unequal_list[i]
                    f_entry=0
                    if j!=c-1 and Diffusion[0][j]=='Break':
                        f_entry=1
                    final=0
                    if i==0 or i==r-1:
                        final=1
            
                    if xk==min(x_unequal_list)  or  f_entry==1:
                        Fin_place(Layers_dict, Terminal_list, AD_obj, transistor, xk, yk, transistor.zk_fin_top, transistor.zk_fin_bottom, final)
                        if final==0:
                            rect=[xk-transistor.width/2-Layers_dict['Poly']['Width']/2, yk-transistor.height/2-Layers_dict['Fin']['Width']/2, -(transistor.depth+transistor.Metal_via_depth)/2, xk+transistor.width/2, yk+transistor.height/2+Layers_dict['Fin']['Width']/2, transistor.zk_fin_bottom]
                            AD_obj.FinFET_regions.append(rect)
                        else:
                            rect=[xk-transistor.width/2-Layers_dict['Poly']['Width']/2, yk-transistor.height_12/2-Layers_dict['Fin']['Width']/2, -(transistor.depth+transistor.Metal_via_depth)/2, xk+transistor.width/2, yk+transistor.height_12/2+Layers_dict['Fin']['Width']/2, transistor.zk_fin_bottom]
                            AD_obj.FinFET_regions.append(rect)
                            
                    elif xk==max(x_unequal_list) or  f_entry==1:
                        Fin_place(Layers_dict, Terminal_list, AD_obj, transistor, xk, yk, transistor.zk_fin_top, transistor.zk_fin_bottom, final)
                        if final==0:
                            rect=[xk-transistor.width/2, yk-transistor.height/2-Layers_dict['Fin']['Width']/2, -(transistor.depth+transistor.Metal_via_depth)/2, xk+transistor.width/2+Layers_dict['Poly']['Width']/2, yk+transistor.height/2+Layers_dict['Fin']['Width']/2, transistor.zk_fin_bottom]
                            AD_obj.FinFET_regions.append(rect)
                        else:
                            rect=[xk-transistor.width/2, yk-transistor.height_12/2-Layers_dict['Fin']['Width']/2, -(transistor.depth+transistor.Metal_via_depth)/2, xk+transistor.width/2+Layers_dict['Poly']['Width']/2, yk+transistor.height_12/2+Layers_dict['Fin']['Width']/2, transistor.zk_fin_bottom]
                            AD_obj.FinFET_regions.append(rect)
                    else:
                        Fin_place(Layers_dict, Terminal_list, AD_obj, transistor, xk, yk, transistor.zk_fin_top, transistor.zk_fin_bottom, final)
                        if final==0:
                            rect=[xk-transistor.width/2, yk-transistor.height/2-Layers_dict['Fin']['Width']/2, -(transistor.depth+transistor.Metal_via_depth)/2, xk+transistor.width/2, yk+transistor.height/2+Layers_dict['Fin']['Width']/2, transistor.zk_fin_bottom]
                            AD_obj.FinFET_regions.append(rect)
                        else:
                            rect=[xk-transistor.width/2, yk-transistor.height_12/2-Layers_dict['Fin']['Width']/2, -(transistor.depth+transistor.Metal_via_depth)/2, xk+transistor.width/2, yk+transistor.height_12/2+Layers_dict['Fin']['Width']/2, transistor.zk_fin_bottom]
                            AD_obj.FinFET_regions.append(rect)
                        
            for i in range (r):
                if i<r-1:
                    if i==0:
                        yk=y_unequal_list[i]
                        cube=[xk-transistor.width/2-Layers_dict['Poly']['Width']/2, yk-transistor.height_12/2-track_num[i]*Layers_dict['Fin']['Pitch']+Layers_dict['Fin']['Width']/2, -(transistor.depth+transistor.Metal_via_depth)/2, -xk+transistor.width/2+Layers_dict['Poly']['Width']/2, yk-transistor.height_12/2-Layers_dict['Fin']['Width']/2, zk_fin_top]
                        AD_obj.Routing_channel_regions.append(cube)    
                    else:
                        yk=y_unequal_list[i]
                        cube=[xk-transistor.width/2-Layers_dict['Poly']['Width']/2, yk-transistor.height/2-track_num[i]*Layers_dict['Fin']['Pitch']+Layers_dict['Fin']['Width']/2, -(transistor.depth+transistor.Metal_via_depth)/2, -xk+transistor.width/2+Layers_dict['Poly']['Width']/2, yk-transistor.height/2-Layers_dict['Fin']['Width']/2, zk_fin_top]
                        AD_obj.Routing_channel_regions.append(cube)    
            
            min_x=min(x_list_r)
            min_y=min(y_list_r)
            
            for i in range (len(obj_tran)):
                k=0
                for j in range (len(obj_tran[i])):
                    obj_tran[i][j].transistor_id['C']=obj_tran[i][j].x
                    obj_tran[i][j].transistor_id['R']=obj_tran[i][j].y
                    obj_tran[i][j].transistor_id['X']=obj_tran[i][j].x_coord
                    obj_tran[i][j].transistor_id['Y']=obj_tran[i][j].y_coord      
                    obj_tran[i][j].transistor_id['Name']=mat_1[i][j]
                    obj_tran[i][j].transistor_id['Left']=S_D_mat_1[i][k]
                    obj_tran[i][j].transistor_id['Right']=S_D_mat_1[i][k+1]
            
            for i in range (r):
                yk=y_unequal_list[i]
                rect=[xk-transistor.width/2, yk-transistor.height/2-Layers_dict['Fin']['Width']/2, -(transistor.depth+transistor.Metal_via_depth)/2, xk+transistor.width/2, yk+transistor.height/2+Layers_dict['Fin']['Width']/2, transistor.zk_fin_bottom]
                AD_obj.FinFET_regions.append(rect) 
            
            yk=(((r+1)/2)-(1))*(transistor.height+transistor.sy)
            x_all_list=[]
            yk=y_unequal_list[0]
            for j in range (c):
                xk=x_unequal_list[j]
                x_all_list.append(xk)
                Gate_port(mat, r, y_unequal_list, j, Layers_dict, Terminal_list, AD_obj, transistor, xk, yk, (transistor.depth+transistor.Metal_via_depth)/2-transistor.Metal_via_depth-transistor.t_gate, (transistor.depth+transistor.Metal_via_depth)/2-transistor.Metal_via_depth)
            
            layout_width=abs(max(x_unequal_list))+transistor.width/2+Layers_dict['Poly']['Width']/2
            layout_height=AD_obj.Poly_regions[0][4]        
    
            CC_plot(layout_width, layout_height, Device_Name, mat_1, r_1, c_1, transistor, transistor.line_w, Layers_dict, obj_tran)    
            
            T_obj=Transient_FDM.Temperature(mat, r, c, Layers_dict, AD_obj, transistor, x_unequal_list, min(x_unequal_list), max(x_unequal_list), min(y_unequal_list), max(y_unequal_list), AD_obj.Poly_regions[0][4], Diffusion_type, obj_tran, transistor.t_max)
            
            plt.savefig(f'Temperature_driven_{AD}_{duty}_{finger_count_list[nf]}_nf.png', dpi=300)
            plt.show()
            
            (t_vector, x_list, y_list, resistance, capacitance, T_time, g_mat, data_list, T_poly_region, RC_circuit)=T_obj.time_solver(transistor.t_max, transistor.time_step)
    
    end_time = time.time()
    
    # Calculate runtime
    runtime = end_time - start_time
    
    # Print the runtime
    print("Runtime:", runtime, "seconds")