#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:52:48 2024

@author: karmo005
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:12:46 2024

@author: karmo005
"""


import numpy as np
from numpy import *
import math

import matplotlib.pyplot as plt

import scipy.sparse as sparse_mat
import scipy.sparse.linalg as sparse_algebra

from collections import Counter


class Element(object):
    def __init__(self, x, y, z):
        self.x=x
        self.y=y
        self.z=z
        self.x_coord=0
        self.y_coord=0
        self.z_coord=0
        self.llx=0
        self.lly=0
        self.llz=0
        self.urx=0
        self.ury=0
        self.urz=0
        self.mark=0
        self.Capacitanceonductivity=0 
        self.material={}
        self.area={}
        self.material_list=[]
        self.Volume_avg={}
        self.Volume_avg_mark=0
        self.material_mark=0
        self.R={}
        self.C={}
        
        self.Resistance=0
        self.Capacitance=0
        self.kt_each=0


        
        

class Temperature:    
    def __init__(self, mat, layout_row, layout_column, Layers_dict, AD_obj, transistor, x_unequal_list, min_x, max_x, min_y, max_y, min_y_layout ,Diffusion_type, obj_tran, t_max):

        self.mat=mat
        self.x_unequal_list=x_unequal_list  
        self.AD_obj=AD_obj
        
        self.layout_row=layout_row
        self.layout_column=layout_column
        self.xk=((1)-((self.layout_column+1)/2))*(transistor.width/2+transistor.sx+Layers_dict['Poly']['Width']/2)
        self.yk=(((self.layout_row+1)/2)-(1))*(transistor.height+transistor.sy)
        self.Diffusion_type=Diffusion_type
        self.t_max=t_max
        self.T_bound=0
        self.obj_tran=obj_tran
        self.data_list=[]
        self.RC=[]
    
        self.transistor_middle=[]
        self.Layers_dict=Layers_dict
        self.transistor=transistor
        self.min_x=min_x
        self.min_y=min_y
        self.max_x = max_x
        self.max_y = max_y
        multiplier=1
        if self.min_x == -self.max_x:
            multiplier=2
            
        self.min_z=(self.transistor.t_substrate+self.transistor.t_box+self.transistor.t_gate)/(2*self.transistor.scale_factor)
        self.layout_height=abs(min_y_layout)
        self.row=int((multiplier*self.layout_height+Layers_dict['Fin']['Pitch']-Layers_dict['Fin']['Width'])/Layers_dict['Fin']['Pitch'])              #GF12


        self.layout_width=abs(self.max_x)+transistor.width/2+Layers_dict['Poly']['Width']/2

        self.column_unequal=0       #GF12
        self.block_width_unequal=0      #GF12
        self.unequal_x_end=0

        self.flag_unequal=self.unequal_or_equal()

        if self.flag_unequal==1:
            self.layout_width=self.layout_width-2*self.unequal_x_end
            
        self.column_equal=int((multiplier*self.layout_width)/10)#GF12        

        self.layer=20
        self.itr=0
        self.cap_value=[]
        self.res_value=[]
        self.T_time=[]

        self.block_width=(multiplier*self.layout_width)/self.column_equal      #GF12
        self.column=self.column_equal+self.column_unequal        

        self.block_height=(multiplier*self.layout_height+Layers_dict['Fin']['Pitch']-Layers_dict['Fin']['Width'])/self.row      #GF12

        self.block_depth=(2*(abs(self.min_z)))/self.layer

        self.depth_num=2
        self.Fin_block_number=math.ceil((self.transistor.zk_fin_top-self.transistor.zk_fin_bottom)/self.block_depth)
        self.Poly_block_number=math.ceil(Layers_dict['Poly']['Width']/self.block_width)

        self.obj=[]
        self.node_number=self.layer*self.row*self.column
        self.Capacitancep = sparse_mat.dok_matrix((self.node_number, self.node_number)) 
        self.P = np.zeros((self.node_number,1))
        self.T = self.T_bound*np.ones((self.node_number,1))
        self.block = []
        self.block_volume=self.block_width*self.block_height*self.block_depth
        self.block_area=self.block_width*self.block_height

        self.x_list=[]
        self.y_list=[]
        self.x_l=[]
        self.y_l=[]
        self.z_list=[]
        
        self.poly_cut=[]
        self.T_poly_region=[]
        self.T_all=[]
        self.T_list=[]
        (self.Fin_count_per_block, self.Poly_count_per_block)=self.Fin_poly_count_per_block()
        #self.block_plot()
        
        if multiplier==2:
            print('are you coming for this')
            self.block_plot()
        else: 
            self.block_plot_unequal()
        self.G = self.G_calculation()

    def find_most_common_value(self, lst):
        count = Counter(lst)

        most_common_value = max(count, key=count.get)
        occurrences = count[most_common_value]
        
        return most_common_value, occurrences

    def unequal_or_equal(self):
        dummy_count=[]
        for i in range (len(self.mat)):
            k=0
            for j in range (len(self.mat[i])):
                if self.mat[i][j]==2:
                    k=k+1
                    if self.mat[i][j+1]==1:
                        break
                    
            dummy_count.append(k)        
        
        flag_unequal=0
        if dummy_count[1]>2:
            flag_unequal=1   
        
            most_dummy_occur, occurance_count = self.find_most_common_value(dummy_count)
            
            most_dummy_occur_2 = most_dummy_occur-2
            
            self.unequal_x_end = self.x_unequal_list[most_dummy_occur_2]
            
            self.column_unequal=int((2*self.unequal_x_end)/40)       #GF12
            self.block_width_unequal=(2*self.unequal_x_end)/self.column_unequal      #GF12
                
        return flag_unequal


    def block_plot_unequal(self):

        for i in range (self.layer): 
            q=[]
            for j in range (self.row):
                p=[]                
                for k in range (self.column):
                    p.append(Element(k, j, i))
                q.append(p)
            self.block.append(q)

    
        for i in range (self.layer): 
            next_lly=-(self.Layers_dict['Fin']['Pitch']-self.Layers_dict['Fin']['Width'])/2
            for j in range (self.row):
                next_llx=0
                next_ury=next_lly+self.block_height
                for k in range (self.column):                
                    
                    self.block[i][j][k].lly=round(next_lly,3)
                    self.block[i][j][k].ury=round(next_ury,3) 
                    
                    self.block[i][j][k].llx=round(next_llx,3)   
                    
                    if self.flag_unequal==1:
                        if k<self.column_unequal//2 or k>self.column_equal+self.column_unequal//2-1:
                            #print('is anything coming here', i, j, k, self.block[i][j][k].llx, right_end, self.unequal_x_end)
                            self.block[i][j][k].urx=round(next_llx+self.block_width_unequal,3) 
                            self.block[i][j][k].x_coord=next_llx+self.block_width_unequal/2
                        else:
                            self.block[i][j][k].urx=round(next_llx+self.block_width,3) 
                            self.block[i][j][k].x_coord=next_llx+self.block_width/2
                    else:                                   
                        self.block[i][j][k].urx=round(next_llx+self.block_width,3) 
                        self.block[i][j][k].x_coord=next_llx+self.block_width/2
                        
                    
                    self.block[i][j][k].y_coord=next_lly+self.block_height/2
                    self.block[i][j][k].z_coord=(((self.layer+1)/2)-(self.block[i][j][k].z+1))*self.block_depth

                    self.block[i][j][k].llz=round(self.block[i][j][k].z_coord-self.block_depth/2,3)
                    self.block[i][j][k].urz=round(self.block[i][j][k].z_coord+self.block_depth/2,3) 

                    if i==0:
                        self.x_list.append(self.block[i][j][k].x_coord)
                        self.y_list.append(self.block[i][j][k].y_coord)
                        self.z_list.append(self.block[i][j][k].z_coord)
                        self.x_l.append(k)
                        self.y_l.append(j) 
                                            
                    next_llx=self.block[i][j][k].urx

                next_lly=next_ury
                


                
        self.block=self.material_per_block()
        self.block=self.R_C_calculation()            

    def block_plot(self):

        for i in range (self.layer): 
            q=[]
            for j in range (self.row):
                p=[]                
                for k in range (self.column):
                    p.append(Element(k, j, i))
                q.append(p)
            self.block.append(q)
        
        for i in range (self.layer):        
            for j in range (self.row):
                for k in range (self.column):                

                    self.block[i][j][k].x_coord=((self.block[i][j][k].x+1)-((self.column+1)/2))*self.block_width
                    self.block[i][j][k].y_coord=(((self.row+1)/2)-(self.block[i][j][k].y+1))*self.block_height
                    self.block[i][j][k].z_coord=(((self.layer+1)/2)-(self.block[i][j][k].z+1))*self.block_depth
                    if i==0:
                        self.x_list.append(self.block[i][j][k].x_coord)
                        self.y_list.append(self.block[i][j][k].y_coord)
                        self.z_list.append(self.block[i][j][k].z_coord)
                        self.x_l.append(k)
                        self.y_l.append(j) 
                    
                    self.block[i][j][k].llx=round(self.block[i][j][k].x_coord-self.block_width/2,3)
                    self.block[i][j][k].lly=round(self.block[i][j][k].y_coord-self.block_height/2,3)
                    self.block[i][j][k].llz=round(self.block[i][j][k].z_coord-self.block_depth/2,3)
                    
                    self.block[i][j][k].urx=round(self.block[i][j][k].x_coord+self.block_width/2,3)
                    self.block[i][j][k].ury=round(self.block[i][j][k].y_coord+self.block_height/2,3) 
                    self.block[i][j][k].urz=round(self.block[i][j][k].z_coord+self.block_depth/2,3) 
                    
        for i in range (1):    
            for j in range (self.row):
                for k in range (self.column):                
                    color_r='lime'
                    line_width=1                    
                    x1=self.block[i][j][k].x_coord-self.block_width/2
                    y1=self.block[i][j][k].y_coord+self.block_height/2
                    x2=self.block[i][j][k].x_coord +self.block_width/2    #top plate
                    y2=self.block[i][j][k].y_coord+self.block_height/2                    
                    plt.plot([x1,x2],[y1,y2], color=color_r, linewidth=line_width) 
                    
                    x3=self.block[i][j][k].x_coord-self.block_width/2
                    y3=self.block[i][j][k].y_coord-self.block_height/2
                    x4=self.block[i][j][k].x_coord +self.block_width/2      #bottom plate
                    y4=self.block[i][j][k].y_coord-self.block_height/2
                    plt.plot([x3,x4],[y3,y4], color=color_r, linewidth=line_width)
                    
                    x1_1=self.block[i][j][k].x_coord-self.block_width/2
                    y1_1=self.block[i][j][k].y_coord+self.block_height/2
                    x2_1=self.block[i][j][k].x_coord-self.block_width/2   #top plate
                    y2_1=self.block[i][j][k].y_coord-self.block_height/2
                    plt.plot([x1_1,x2_1],[y1_1,y2_1], color=color_r, linewidth=line_width) 
                    
                    x3_1=self.block[i][j][k].x_coord +self.block_width/2  
                    y3_1=self.block[i][j][k].y_coord+self.block_height/2
                    x4_1=self.block[i][j][k].x_coord +self.block_width/2      #bottom plate
                    y4_1=self.block[i][j][k].y_coord-self.block_height/2
                    plt.plot([x3_1,x4_1],[y3_1,y4_1], color=color_r, linewidth=line_width)

        self.block=self.material_per_block()
        self.block=self.R_calculation()



    def Fin_poly_count_per_block(self):
        Fin_count_per_block=1
        Poly_count_per_block=1
        if self.Layers_dict['Fin']['Width']<self.block_height: 
            if self.block_height>self.Layers_dict['Fin']['Pitch']:
                Fin_count_per_block=self.block_height/self.Layers_dict['Fin']['Pitch']
        
        if self.Layers_dict['Poly']['Width']<self.block_width: 
            if self.block_height>self.Layers_dict['Poly']['Pitch']:
                Poly_count_per_block=self.block_height/self.Layers_dict['Poly']['Pitch']
           
        return Fin_count_per_block, Poly_count_per_block
        
    def poly_region_find(self, i, j, k):
        flag=0
        for p in range (len(self.AD_obj.Poly_regions)):
            if self.AD_obj.Poly_regions[p][2]<=self.block[i][j][k].urz<=self.AD_obj.Poly_regions[p][5] or self.AD_obj.Poly_regions[p][2]<=self.block[i][j][k].llz<=self.AD_obj.Poly_regions[p][5] or self.block[i][j][k].llz<=self.AD_obj.Poly_regions[p][2]<=self.block[i][j][k].urz or self.block[i][j][k].llz<=self.AD_obj.Poly_regions[p][5]<=self.block[i][j][k].urz:
                if self.block[i][j][k].lly<=self.AD_obj.Poly_regions[p][1]<=self.block[i][j][k].ury or self.block[i][j][k].lly<=self.AD_obj.Poly_regions[p][4]<=self.block[i][j][k].ury or self.AD_obj.Poly_regions[p][1]<=self.block[i][j][k].lly<=self.block[i][j][k].ury<=self.AD_obj.Poly_regions[p][4]:
                    if self.AD_obj.Poly_regions[p][0]<=self.block[i][j][k].llx<self.AD_obj.Poly_regions[p][3] or self.AD_obj.Poly_regions[p][0]<self.block[i][j][k].urx<=self.AD_obj.Poly_regions[p][3]:                            
                        self.block[i][j][k].material['Poly']=self.AD_obj.Poly_regions[p]
                        self.block[i][j][k].material_list.append('Poly')
                        flag=1                    
                        break
                    
        return flag

    def Metal_via_region(self, i, j, k, region, var):
        flag=0
        flag_2=0
        for p in range (len(region)):
            if self.block[i][j][k].llz<=region[p][2]<=self.block[i][j][k].urz or self.block[i][j][k].llz<=region[p][5]<=self.block[i][j][k].urz or region[p][2]<=self.block[i][j][k].llz<region[p][5] or region[p][2]<self.block[i][j][k].urz<=region[p][5]:
                if self.block[i][j][k].lly<=region[p][1]<=self.block[i][j][k].ury or self.block[i][j][k].lly<=region[p][4]<=self.block[i][j][k].ury or region[p][1]<=self.block[i][j][k].lly<=region[p][4] or region[p][1]<=self.block[i][j][k].ury<=region[p][4]:
                    if self.block[i][j][k].llx<region[p][0]<self.block[i][j][k].urx or self.block[i][j][k].llx<region[p][3]<self.block[i][j][k].urx or region[p][0]<self.block[i][j][k].llx<region[p][3] or region[p][0]<self.block[i][j][k].urx<region[p][3]:                        
                        self.block[i][j][k].material[var]=region[p]
                        self.block[i][j][k].material_list.append(var)

                        flag=1
                        flag_2=1
                        
                            
                        break 

        dy=0
        dx=0
        if flag==1:
            if self.block[i][j][k].lly<=self.block[i][j][k].material[var][1]<self.block[i][j][k].ury :
                dy=self.block[i][j][k].ury-self.block[i][j][k].material[var][1]
            elif self.block[i][j][k].lly<self.block[i][j][k].material[var][4]<=self.block[i][j][k].ury :
                dy=self.block[i][j][k].material[var][4]-self.block[i][j][k].lly
            elif self.block[i][j][k].material[var][1]<=self.block[i][j][k].lly<=self.block[i][j][k].ury<=self.block[i][j][k].material[var][4] :
                dy=self.block[i][j][k].ury-self.block[i][j][k].lly                                    


            if self.block[i][j][k].llx<=self.block[i][j][k].material[var][0]<self.block[i][j][k].urx :
                dx=self.block[i][j][k].urx-self.block[i][j][k].material[var][0]
            elif self.block[i][j][k].llx<self.block[i][j][k].material[var][3]<=self.block[i][j][k].urx :
                dx=self.block[i][j][k].material[var][3]-self.block[i][j][k].llx
            elif self.block[i][j][k].material[var][0]<=self.block[i][j][k].llx<=self.block[i][j][k].urx<=self.block[i][j][k].material[var][3]:
                dx=self.block[i][j][k].urx-self.block[i][j][k].llx
            elif self.block[i][j][k].llx<=self.block[i][j][k].material[var][0]<=self.block[i][j][k].material[var][3]<=self.block[i][j][k].urx:
                dx=self.block[i][j][k].material[var][3]-self.block[i][j][k].material[var][0]  

                                        
                                    
            M1_area=dx*dy
            
            SiO2_area=0
            SiO2_area=self.block_area-M1_area
            self.block[i][j][k].R[var]=self.block_depth/(self.transistor.thermal_conductivity['contact']*M1_area)
            self.block[i][j][k].Volume_avg[var]=(M1_area*self.block_depth)/self.block_volume
                   
            
            if SiO2_area!=0:
                self.block[i][j][k].R['SiO2']=self.block_depth/(self.transistor.thermal_conductivity['SiO2']*SiO2_area)
                self.block[i][j][k].material_list.append('SiO2')
                
 

        if flag==0:
            self.block[i][j][k].R['SiO2']=self.block_depth/(self.transistor.thermal_conductivity['SiO2']*self.block_area) 
            self.block[i][j][k].material_list.append('SiO2')
            flag=1

        return flag, flag_2

    def Fin_u_Poly(self, i, j):
        flag=0
        k_v=0
        for k in range (len(self.block[i][j])):
            if 'Fin' in self.block[i][j][k].material_list:
                k_v=k
                flag=1
                break
        return flag , k_v 

    def area_calculation(self, var, i, j ,k):
        dy=0
        dx=0

        if self.block[i][j][k].lly<=self.block[i][j][k].material[var][1]<self.block[i][j][k].ury :
            dy=self.block[i][j][k].ury-self.block[i][j][k].material[var][1]
        elif self.block[i][j][k].lly<self.block[i][j][k].material[var][4]<=self.block[i][j][k].ury :
            dy=self.block[i][j][k].material[var][4]-self.block[i][j][k].lly
        elif self.block[i][j][k].material[var][1]<=self.block[i][j][k].lly<=self.block[i][j][k].ury<=self.block[i][j][k].material[var][4] :
            dy=self.block[i][j][k].ury-self.block[i][j][k].lly                                    
        elif self.block[i][j][k].lly<=self.block[i][j][k].material[var][1]<=self.block[i][j][k].material[var][4]<=self.block[i][j][k].ury :
            dy=self.block[i][j][k].material[var][4]-self.block[i][j][k].material[var][1]  

        if self.block[i][j][k].llx<=self.block[i][j][k].material[var][0]<self.block[i][j][k].urx :
            dx=self.block[i][j][k].urx-self.block[i][j][k].material[var][0]
        elif self.block[i][j][k].llx<self.block[i][j][k].material[var][3]<=self.block[i][j][k].urx :
            dx=self.block[i][j][k].material[var][3]-self.block[i][j][k].llx
        elif self.block[i][j][k].material[var][0]<=self.block[i][j][k].llx<=self.block[i][j][k].urx<=self.block[i][j][k].material[var][3]:
            dx=self.block[i][j][k].urx-self.block[i][j][k].llx
        elif self.block[i][j][k].llx<=self.block[i][j][k].material[var][0]<=self.block[i][j][k].material[var][3]<=self.block[i][j][k].urx:
            dx=self.block[i][j][k].material[var][3]-self.block[i][j][k].material[var][0]  
        
        return dx*dy


    def SiO2_region(self, i, j, k, var):
        multiplier=1
        if var=='Fin' and self.Fin_count_per_block>0:
            multiplier=self.Fin_count_per_block
        if var=='Poly' and self.Poly_count_per_block>0:
            multiplier=self.Poly_count_per_block
            
        self.block[i][j][k].area['SiO2']=self.block_area-multiplier*self.block[i][j][k].area[var]
        unit_change=10**(-27)
        if self.block[i][j][k].area['SiO2']!=0:
            self.kt_cal('SiO2')
            self.block[i][j][k].R['SiO2']=self.block_depth/(self.transistor.thermal_conductivity['SiO2']*self.block[i][j][k].area['SiO2'])
            self.block[i][j][k].C['SiO2']=self.transistor.heat_capacity['SiO2']*self.transistor.density['SiO2']*unit_change*self.block[i][j][k].area['SiO2']*self.block_depth

            self.block[i][j][k].material_list.append('SiO2')
            
    def material_per_block(self):
        unit_change=10**(-27)
        for i in range (len(self.block)):
            for j in range (len(self.block[i])):
                for k in range (len(self.block[i][j])):                        

                    if self.block[i][j][k].material_mark==0 :               
                        flag_1=0

                        if self.transistor.zk_fin_top<=self.block[i][j][k].urz:
                            (flag_1, flag_2)=self.Metal_via_region(i, j, k, self.AD_obj.M1_regions, 'M1')
                            
                            
                            if flag_2==0:  
                                (flag_1, flag_2)=self.Metal_via_region(i, j, k, self.AD_obj.V0_regions, 'V0')                            
                            if flag_2==0:
                                (flag_1, flag_2)=self.Metal_via_region(i, j, k, self.AD_obj.M2_regions, 'M2')
                            if flag_2==0:
                                (flag_1, flag_2)=self.Metal_via_region(i, j, k, self.AD_obj.V1_regions, 'V1')                                
                            if flag_2==0:
                                (flag_1, flag_2)=self.Metal_via_region(i, j, k, self.AD_obj.M3_regions, 'M3')
                            
                        if 'M1' not in self.block[i][j][k].material_list:
                            
                            if  self.block[i][j][k].urz<=(self.transistor.depth+self.transistor.Metal_via_depth)/2-self.transistor.Metal_via_depth or self.block[i][j][k].llz<=(self.transistor.depth+self.transistor.Metal_via_depth)/2-self.transistor.Metal_via_depth<=self.block[i][j][k].urz: 
                                                
                                for p in range (len(self.AD_obj.Fin_regions)):
                                    if self.block[i][j][k].llx<=self.AD_obj.Fin_regions[p][0]<self.block[i][j][k].urx or self.block[i][j][k].llx<self.AD_obj.Fin_regions[p][3]<=self.block[i][j][k].urx or self.AD_obj.Fin_regions[p][0]<=self.block[i][j][k].llx<=self.block[i][j][k].urx :
                                        if  self.block[i][j][k].lly<self.AD_obj.Fin_regions[p][1]<self.block[i][j][k].ury or self.block[i][j][k].lly<self.AD_obj.Fin_regions[p][4]<self.block[i][j][k].ury or self.AD_obj.Fin_regions[p][1]<self.block[i][j][k].lly<self.AD_obj.Fin_regions[p][4] or self.AD_obj.Fin_regions[p][1]<self.block[i][j][k].ury<self.AD_obj.Fin_regions[p][4]:
                                            flag=self.poly_region_find(i, j, k)

                                            if flag==0:
                                                if self.block[i][j][k].llz<=self.AD_obj.Fin_regions[p][2]<=self.block[i][j][k].urz or self.block[i][j][k].llz<=self.AD_obj.Fin_regions[p][5]<=self.block[i][j][k].urz or self.AD_obj.Fin_regions[p][2]<=self.block[i][j][k].llz<=self.AD_obj.Fin_regions[p][5] or self.AD_obj.Fin_regions[p][2]<=self.block[i][j][k].urz<=self.AD_obj.Fin_regions[p][5] :

                                                    self.block[i][j][k].material['Fin']=self.AD_obj.Fin_regions[p]
                                                    self.block[i][j][k].material_list.append('Fin')

                                                    self.block[i][j][k].area['Fin']=self.area_calculation('Fin', i, j ,k)
                                                    
                                                    if self.block[i][j][k].area['Fin']!=self.block_area:
                                                        self.SiO2_region(i, j, k, 'Fin')

                                                    self.block[i][j][k].R['Fin']=self.block_depth/(self.kt_cal('Fin')*self.block[i][j][k].area['Fin'])
                                                    self.block[i][j][k].C['Fin']=self.transistor.heat_capacity['Si']*self.transistor.density['Si']*unit_change*self.block[i][j][k].area['Fin']*self.block_depth

                                                    if self.Fin_count_per_block>1:
                                                        self.block[i][j][k].R['Fin']=(self.block_depth/(self.kt_cal('Fin')*self.block[i][j][k].area['Fin']))/self.Fin_count_per_block
                                                        self.block[i][j][k].C['Fin']=self.Fin_count_per_block*self.transistor.heat_capacity['Si']*self.transistor.density['Si']*unit_change*self.block[i][j][k].area['Fin']*self.block_depth
                                                    break
                
                            
                                for p in range (len(self.AD_obj.Poly_regions)):                 
                                    if self.AD_obj.Poly_regions[p][2]<=self.block[i][j][k].urz<=self.AD_obj.Poly_regions[p][5] or self.AD_obj.Poly_regions[p][2]<=self.block[i][j][k].llz<=self.AD_obj.Poly_regions[p][5] or self.block[i][j][k].llz<=self.AD_obj.Poly_regions[p][2]<=self.block[i][j][k].urz or self.block[i][j][k].llz<=self.AD_obj.Poly_regions[p][5]<=self.block[i][j][k].urz:
                 
                                        if self.block[i][j][k].lly<=self.AD_obj.Poly_regions[p][1]<self.block[i][j][k].ury or self.block[i][j][k].lly<self.AD_obj.Poly_regions[p][4]<=self.block[i][j][k].ury or self.block[i][j][k].lly<=self.AD_obj.Poly_regions[p][4]<=self.block[i][j][k].ury or self.AD_obj.Poly_regions[p][1]<=self.block[i][j][k].lly<=self.block[i][j][k].ury<=self.AD_obj.Poly_regions[p][4]:
                                            if self.block[i][j][k].llx<self.AD_obj.Poly_regions[p][0]<self.block[i][j][k].urx or self.block[i][j][k].llx<self.AD_obj.Poly_regions[p][3]<self.block[i][j][k].urx or self.AD_obj.Poly_regions[p][0]<self.block[i][j][k].llx<self.AD_obj.Poly_regions[p][3] or self.AD_obj.Poly_regions[p][0]<self.block[i][j][k].urx<self.AD_obj.Poly_regions[p][3]:
                                                (Fin_flag, k_v)=self.Fin_u_Poly(i, j)

                                                if Fin_flag==1:
                                                    self.block[i][j][k].material['Fin_u']=self.AD_obj.Poly_regions[p]
                                                    self.block[i][j][k].material_list.append('Fin_u')
                                                    self.block[i][j][k].material['Fin']=self.block[i][j][k_v].material['Fin']                                                

                                                self.block[i][j][k].material['Poly']=self.AD_obj.Poly_regions[p]
                                                self.block[i][j][k].material_list.append('Poly')
                                                # for dummy in range (len(self.AD_obj.Dummy_regions)):
                                                #     if self.AD_obj.Dummy_regions[dummy]==self.AD_obj.Poly_regions[p]:
                                                #         self.block[i][j][k].material_list.append('Dummy')
                                                #         break
                                                
                                                self.block[i][j][k].area['Poly']=self.area_calculation('Poly', i, j ,k)
                                                if self.block[i][j][k].area['Poly']!=self.block_area:
                                                    self.SiO2_region(i, j, k, 'Poly')

                                                self.block[i][j][k].R['Poly']=self.block_depth/(self.kt_cal('Poly')*self.block[i][j][k].area['Poly'])
                                                self.block[i][j][k].C['Poly']=self.transistor.heat_capacity['gate']*self.transistor.density['gate']*unit_change*self.block[i][j][k].area['Poly']*self.block_depth

                                                if self.Poly_count_per_block>1:
                                                    self.block[i][j][k].R['Poly']=(self.block_depth/(self.kt_cal('Poly')*self.block[i][j][k].area['Poly']))/self.Poly_count_per_block
                                                    self.block[i][j][k].C['Poly']=self.Poly_count_per_block*self.transistor.heat_capacity['gate']*self.transistor.density['gate']*unit_change*self.block[i][j][k].area['Poly']*self.block_depth
                                                    
                                                break  
                                            
                                for p in range (len(self.AD_obj.Dummy_regions)):                 
                                    if self.AD_obj.Dummy_regions[p][2]<=self.block[i][j][k].urz<=self.AD_obj.Dummy_regions[p][5] or self.AD_obj.Dummy_regions[p][2]<=self.block[i][j][k].llz<=self.AD_obj.Dummy_regions[p][5] or self.block[i][j][k].llz<=self.AD_obj.Dummy_regions[p][2]<=self.block[i][j][k].urz or self.block[i][j][k].llz<=self.AD_obj.Dummy_regions[p][5]<=self.block[i][j][k].urz:
                 
                                        if self.block[i][j][k].lly<=self.AD_obj.Dummy_regions[p][1]<self.block[i][j][k].ury or self.block[i][j][k].lly<self.AD_obj.Dummy_regions[p][4]<=self.block[i][j][k].ury or self.block[i][j][k].lly<=self.AD_obj.Dummy_regions[p][4]<=self.block[i][j][k].ury or self.AD_obj.Dummy_regions[p][1]<=self.block[i][j][k].lly<=self.block[i][j][k].ury<=self.AD_obj.Dummy_regions[p][4]:
                                            if self.block[i][j][k].llx<self.AD_obj.Dummy_regions[p][0]<self.block[i][j][k].urx or self.block[i][j][k].llx<self.AD_obj.Dummy_regions[p][3]<self.block[i][j][k].urx or self.AD_obj.Dummy_regions[p][0]<self.block[i][j][k].llx<self.AD_obj.Dummy_regions[p][3] or self.AD_obj.Dummy_regions[p][0]<self.block[i][j][k].urx<self.AD_obj.Dummy_regions[p][3]:
        
                                                self.block[i][j][k].material['Dummy']=self.AD_obj.Dummy_regions[p]
                                                self.block[i][j][k].material_list.append('Dummy')
                                                    
                                                break                                          

                                for p in range (len(self.AD_obj.FinFET_regions)):
                                    if self.block[i][j][k].llx<=self.AD_obj.FinFET_regions[p][0]<self.block[i][j][k].urx or self.block[i][j][k].llx<self.AD_obj.FinFET_regions[p][3]<=self.block[i][j][k].urx or self.AD_obj.FinFET_regions[p][0]<=self.block[i][j][k].llx<=self.block[i][j][k].urx :    
                                        if self.block[i][j][k].llz<=self.AD_obj.FinFET_regions[p][5]<=self.block[i][j][k].urz or self.block[i][j][k].llz<=self.block[i][j][k].urz<=self.AD_obj.FinFET_regions[p][5]:
                                            if self.block[i][j][k].lly<self.AD_obj.FinFET_regions[p][4]<=self.block[i][j][k].ury or self.AD_obj.FinFET_regions[p][1]<=self.block[i][j][k].lly<self.AD_obj.FinFET_regions[p][4] or self.AD_obj.FinFET_regions[p][1]<self.block[i][j][k].ury<=self.AD_obj.FinFET_regions[p][4]:
    
                                                self.block[i][j][k].material['Subs']=self.AD_obj.FinFET_regions[p]
                                                self.block[i][j][k].material_list.append('Subs')
    
                                                self.block[i][j][k].area['Subs']=self.area_calculation('Subs', i, j ,k)
                                                if self.block[i][j][k].area['Subs']!=self.block_area:
                                                    self.SiO2_region(i, j, k, 'Subs')
                                                
                                                
                                                self.block[i][j][k].R['Subs']=self.block_depth/(self.kt_cal('Subs')*self.block[i][j][k].area['Subs'])
                                                self.block[i][j][k].C['Subs']=self.transistor.heat_capacity['Si']*self.transistor.density['Si']*unit_change*self.block[i][j][k].area['Subs']*self.block_depth
    
                                                break
                        
                    
                    if  'Subs' not in self.block[i][j][k].material_list and 'Fin' not in self.block[i][j][k].material_list and 'Poly' not in self.block[i][j][k].material_list and 'M1' not in self.block[i][j][k].material_list and 'M2' not in self.block[i][j][k].material_list and 'M3' not in self.block[i][j][k].material_list and 'V0' not in self.block[i][j][k].material_list and 'V1' not in self.block[i][j][k].material_list:

                        self.block[i][j][k].R['SiO2']=self.block_depth/(self.transistor.thermal_conductivity['SiO2']*self.block_area)
                        self.block[i][j][k].area['SiO2']=self.block_area
                        
                        self.block[i][j][k].C['SiO2']=self.transistor.heat_capacity['SiO2']*self.transistor.density['SiO2']*unit_change*self.block[i][j][k].area['SiO2']*self.block_depth


                        self.block[i][j][k].material_list.append('SiO2')
                    
                    
                    self.block[i][j][k].material_mark=1 

                        
                    if 'Poly' in self.block[i][j][k-1].material_list and 'Poly' in self.block[i][j][k].material_list:
                        self.poly_cut.append(k-1)
                        self.poly_cut.append(k)
        #print('block_depth', self.block_depth, self.block_height, self.block_width)

        return self.block




    def kt_cal(self, var):
        kt=0
        if var=='Fin':
            kt=self.transistor.thermal_conductivity['Si NMOS fin']
        elif var=='Poly':
            kt=self.transistor.thermal_conductivity['gate']
        elif var=='Subs':
            kt=self.transistor.thermal_conductivity['Si NMOS SD']
        elif var=='SiO2':
            kt=self.transistor.thermal_conductivity['SiO2']
        elif var=='M1':
            kt=self.transistor.thermal_conductivity['Si NMOS SD']
        elif var=='M2':
            kt=self.transistor.thermal_conductivity['Si NMOS SD']
        elif var=='M3':
            kt=self.transistor.thermal_conductivity['Si NMOS SD']
        elif var=='V0':
            kt=self.transistor.thermal_conductivity['contact']
        elif var=='V1':
            kt=self.transistor.thermal_conductivity['contact']             
            
        return kt


    def R_C_calculation(self):
        for i in range (len(self.block)):
            for j in range (len(self.block[i])):
                for k in range (len(self.block[i][j])):
                    res=[]
                    cap=[]
                    if 'Poly' in self.block[i][j][k].material_list:
                        res.append(abs(self.block[i][j][k].R['Poly']))
                        cap.append(abs(self.block[i][j][k].C['Poly']))
                        
                    if 'SiO2' in self.block[i][j][k].material_list:
                        res.append(abs(self.block[i][j][k].R['SiO2']))
                        cap.append(abs(self.block[i][j][k].C['SiO2']))

                    if 'Fin' in self.block[i][j][k].material_list:                        
                        res.append(abs(self.block[i][j][k].R['Fin']))
                        cap.append(abs(self.block[i][j][k].C['Fin']))
                    
                    if 'Subs' in self.block[i][j][k].material_list:                        
                        res.append(abs(self.block[i][j][k].R['Subs']))
                        cap.append(abs(self.block[i][j][k].C['Subs']))

                    if 'M1' in self.block[i][j][k].material_list:
                        res.append(abs(self.block[i][j][k].R['M1'])) 
                        cap.append(abs(self.block[i][j][k].C['M1']))
                        
                    if 'M2' in self.block[i][j][k].material_list:
                        res.append(abs(self.block[i][j][k].R['M2']))
                        cap.append(abs(self.block[i][j][k].C['M2']))

                    if 'M3' in self.block[i][j][k].material_list:
                        res.append(abs(self.block[i][j][k].R['M3']))
                        cap.append(abs(self.block[i][j][k].C['M3']))

                    if 'V0' in self.block[i][j][k].material_list:
                        res.append(abs(self.block[i][j][k].R['V0'])) 
                        cap.append(abs(self.block[i][j][k].C['V0']))

                    if 'V1' in self.block[i][j][k].material_list:
                        res.append(abs(self.block[i][j][k].R['V1'])) 
                        cap.append(abs(self.block[i][j][k].C['V1']))

                    if 'SiO2' in self.block[i][j][k].material_list:
                        res.append(abs(self.block[i][j][k].R['SiO2']))
                        cap.append(abs(self.block[i][j][k].C['SiO2']))                         
                    

                    self.block[i][j][k].Resistance=res[0]
                    for l in range (1, len(res)):
                        self.block[i][j][k].Resistance=1/(1/self.block[i][j][k].Resistance+1/res[l])
                    self.res_value.append(self.block[i][j][k].Resistance)
                    
                    self.block[i][j][k].Capacitance=cap[0]
                    for l in range (1, len(cap)):
                        self.block[i][j][k].Capacitance=cap[l]+self.block[i][j][k].Capacitance
                    self.cap_value.append(self.block[i][j][k].Capacitance)
            
            rc_dict={}
            rc_dict['Res']=self.block[i][7][18].Resistance
            rc_dict['Cap']=self.block[i][7][18].Capacitance
            self.RC.append(rc_dict)
                    
        return self.block                                               

    def Resistance_equivalent(self, i, j, k, p, q, r): 
        return (self.block[k][j][i].Resistance+self.block[r][q][p].Resistance)/2       
    
    def node_index(self, k, j, i):
        index=0
        if (i < 0 or i >= self.layer or j < 0 or j >= self.row or k < 0 or k >= self.column):
            index = -1
        else:
            index = self.row * self.column * i + j * self.column + k
        return index          

    def substrate_middle_block(self, k, j, i, flag_3):
        for p in range (len(self.obj_tran)):
            for q in range (len(self.obj_tran[p])):
                enter=0
                    
                if flag_3==1 and self.obj_tran[p][q].transistor_id['Y']-self.transistor.height/2<=self.block[i][j][k].lly<=self.block[i][j][k].ury<=self.obj_tran[p][q].transistor_id['Y']+self.transistor.height/2:
                    enter=1
                
                elif flag_3==1 and self.block[i][j][k].lly<=self.obj_tran[p][q].transistor_id['Y']-self.transistor.height/2<=self.block[i][j][k].ury:
                    enter=1
                    
                elif flag_3==1 and self.block[i][j][k].lly<=self.obj_tran[p][q].transistor_id['Y']+self.transistor.height/2<=self.block[i][j][k].ury:
                    enter=1                     
                
                if self.block[i][j][k].llx<=self.obj_tran[p][q].transistor_id['X']<=self.block[i][j][k].urx and enter==1:
                    transistor_mid={}
                    transistor_mid['R']=self.obj_tran[p][q].transistor_id['R']
                    transistor_mid['C']=self.obj_tran[p][q].transistor_id['C']
                    transistor_mid['X']=self.obj_tran[p][q].transistor_id['X']
                    transistor_mid['Y']=self.obj_tran[p][q].transistor_id['Y']                    
                    transistor_mid['Name']=self.obj_tran[p][q].transistor_id['Name']
                    transistor_mid['Res']=self.block[i][j][k].Resistance
                    transistor_mid['Cap']=self.block[i][j][k].Capacitance              
                    transistor_mid['i']=i
                    transistor_mid['j']=j
                    transistor_mid['k']=k 
                    if 'Fin' in self.block[i][j][k].material_list or 'Fin_u' in self.block[i][j][k].material_list:
                        transistor_mid['Fin']=self.block[i][j][k].material['Fin']
                    self.transistor_middle.append(transistor_mid)
                    
                    if 'Poly' in self.block[i][j][k].material_list:
                        self.T_poly_region.append(transistor_mid)
                    
                    break
        return self.transistor_middle, self.T_poly_region    
    

    def G_calculation(self):
        self.G=sparse_mat.dok_matrix((self.layer*self.row*self.column, self.layer*self.row*self.column)) 

        for i in range (self.layer):
            for j in range (self.row):
                for k in range (self.column):
                    flag_3=0
                    if 'Fin' in self.block[i][j][k].material_list:
                        flag_3=1
                    
                    if 'Fin_u' in self.block[i][j][k].material_list:
                        flag_3=1 
                        
                    if flag_3==1: 
                        (self.transistor_middle, self.T_poly_region)=self.substrate_middle_block(k, j, i, flag_3)
                        
                    cur_node = self.node_index(k, j, i)
                    self.G[cur_node, cur_node] = self.block[i][j][k].Capacitance/self.transistor.time_step
                    #+ve x
                    next_node = self.node_index(k + 1, j, i)
                    if next_node >= 0:
                        cond=1/(self.Resistance_equivalent(k, j, i, k+1, j, i))

                        self.G[cur_node, next_node] = -cond
                        self.G[cur_node, cur_node] = self.G[cur_node, cur_node] + cond 
                    #-ve x
                    next_node = self.node_index(k - 1, j, i)
                    if next_node >= 0:
                        cond=1/(self.Resistance_equivalent(k, j, i, k-1, j, i))
                        
                        self.G[cur_node, next_node] = -cond
                        self.G[cur_node, cur_node] = self.G[cur_node, cur_node] + cond 
                    #+ve y
                    next_node = self.node_index(k, j + 1, i)
                    if next_node >= 0:
                        cond=1/(self.Resistance_equivalent(k, j, i, k, j+1, i))

                        self.G[cur_node, next_node] = -cond
                        self.G[cur_node, cur_node] = self.G[cur_node, cur_node] + cond 
                    #-ve y
                    next_node = self.node_index( k, j - 1, i)
                    if next_node >= 0:
                        cond=1/(self.Resistance_equivalent(k, j, i, k, j-1, i))

                        self.G[cur_node, next_node] = -cond
                        self.G[cur_node, cur_node] = self.G[cur_node, cur_node] + cond
                        
                    #+ve z
                    next_node = self.node_index(k, j, i + 1)
                    if next_node >= 0:
                        cond=1/(self.Resistance_equivalent(k, j, i, k, j, i+1))                 

                        self.G[cur_node, next_node] = -cond
                        self.G[cur_node, cur_node] = self.G[cur_node, cur_node] + cond

                    else:  
                        kt=self.transistor.thermal_conductivity['contact']
                        cond=(kt*self.block_width*self.block_height)/self.transistor.t_cnt2gnd
                        self.G[cur_node, cur_node] = self.G[cur_node, cur_node] + cond
                    #-ve z
                    next_node = self.node_index(k, j, i - 1)
                    if next_node >= 0:
                        cond=1/(self.Resistance_equivalent(k, j, i, k, j, i-1))

                        self.G[cur_node, next_node] = -cond
                        self.G[cur_node, cur_node] = self.G[cur_node, cur_node] + cond
                    else:
                        kt=self.transistor.thermal_conductivity['contact']
                        cond=(kt*self.block_width*self.block_height)/self.transistor.t_sub2gnd
                        self.G[cur_node, cur_node] = self.G[cur_node, cur_node] + cond                            

        return self.G

    def P_vector(self, t, T, flag, time_count):

        node_count=0
        T_time_i=[]
        data_dict={}
        
        time_list=[]
        x_c_list=[]
        y_c_list=[]
        P_list=[]
        T_list=[]
        
        for i in range (self.layer):
            for j in range (self.row):
                for k in range (self.column):
                    self.P[node_count]=(self.block[i][j][k].Capacitance/self.transistor.time_step)*self.T[node_count][0]                     
                    T_time_i.append(self.T[node_count][0])
                    current=self.transistor.current
                    if 'Fin_u' in self.block[i][j][k].material_list and 'Dummy'  not in self.block[i][j][k].material_list and 'Poly' in self.block[i][j][k].material_list  and self.min_x+self.transistor.width/2-self.transistor.poly_width <self.block[i][j][k].x_coord< self.max_x-self.transistor.width/2+self.transistor.poly_width  and self.min_y+self.transistor.height/2+self.transistor.sy <self.block[i][j][k].y_coord< self.max_y-self.transistor.height/2-self.transistor.sy and flag==0:
                        self.P[node_count]=self.P[node_count]+(current/(self.depth_num*self.transistor.Fin_number*self.Poly_block_number))*self.transistor.voltage                        
                    
                    if i==1:
                        time_list.append(time_count)
                        x_c_list.append(self.block[i][j][k].x)
                        y_c_list.append(self.block[i][j][k].y)
                        P_list.append(self.P[node_count][0])
                        T_list.append(self.T[node_count][0])                    
                    
                    node_count+=1
                    
        data_dict['t']=time_list
        data_dict['x']=x_c_list
        data_dict['y']=y_c_list
        data_dict['P']=P_list
        data_dict['T']=T_list
        self.T_time.append(T_time_i)

        return self.P, T_time_i, data_dict


    def setTemp(self, T):
        self.T = T

    def T_vector(self, P):
        G = self.G.tocsc()
        
        P = sparse_mat.csc_matrix(self.P)
        I = sparse_mat.identity(G.shape[0]) * 1e-15
        G = G + I 

        self.T = sparse_algebra.spsolve(G, P, permc_spec=None, use_umfpack=True) 
        self.T = self.T.reshape((self.layer*self.column*self.row, 1), order='F')        
        
        return self.T
                                

    def time_solver(self, t_max, time_step):
        prev_t=0
        time_count=0
        flag=0
        
        for i in np.arange(0, t_max, time_step):           
            flag=0

            if prev_t+self.transistor.t_on==i or prev_t+self.transistor.t_on<=i<=prev_t+self.transistor.t_on+self.transistor.t_off:
                if prev_t+self.transistor.t_on==i or prev_t+self.transistor.t_on<=i<prev_t+self.transistor.t_on+self.transistor.t_off:
                    flag=1

                if i==prev_t+self.transistor.t_on+self.transistor.t_off:
                    prev_t=i

            (self.P, T_time_i, data_dict)=self.P_vector(i, self.T, flag, time_count)
            time_count+=1
            print(f'time: {round(i,10)}, T: {max(T_time_i)}')
            self.T_all.append(self.T)
            self.T=self.T_vector(self.P)
            
            self.data_list.append(data_dict)
            
        self.T_all.append(self.T)
            
        return self.T_all, self.x_list, self.y_list, self.res_value, self.cap_value, self.T_time, self.G, self.data_list, self.T_poly_region, self.RC
        
 