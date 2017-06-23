"""
Created on Tue May  9 15:33:53 2017

Author: Amanda
"""

mg0, z0, s0, oid = parabolic_grid(s = 0.0001)
vol = model_run(mg0, z0, s0, oid, storm_duration = 7200, model_run_time = 12000, rainfall_mmhr = 100)
print(vol)

mg1, z1, s1, oid = parabolic_grid(s = 0.0005)
vol1 = model_run(mg1, z1, s1, oid, storm_duration = 7200, model_run_time = 12000, rainfall_mmhr = 100)
print(vol1)

mg2, z2, s2, oid = parabolic_grid(s = 0.001)
vol2 = model_run(mg2, z2, s2, oid, storm_duration = 7200, model_run_time = 12000, rainfall_mmhr = 100)
print(vol2)

mg3, z3, s3, oid = parabolic_grid(s = 0.005)
vol3 = model_run(mg3, z3, s3, oid, storm_duration = 7200, model_run_time = 12000, rainfall_mmhr = 100)
print(vol3)

mg4, z4, s4, oid= parabolic_grid(s = 0.01)
vol4 = model_run(mg4, z4, s4, oid, storm_duration = 7200, model_run_time = 12000, rainfall_mmhr = 100)
print(vol4)

mg5, z5, s5, oid = parabolic_grid(s = 0.02)
vol5 = model_run(mg5, z5, s5, oid, storm_duration = 7200, model_run_time = 12000, rainfall_mmhr = 100)
print(vol5)

mg6, z6, s6, oid = parabolic_grid(s = 0.03)
vol6 = model_run(mg6, z6, s6, oid, storm_duration = 7200, model_run_time = 12000, rainfall_mmhr = 100)
print(vol6)

mg7, z7, s7, oid = parabolic_grid(s = 0.04)
vol7 = model_run(mg7, z7, s7, oid, storm_duration = 7200, model_run_time = 12000, rainfall_mmhr = 100)
print(vol7)

mg8, z8, s8, oid = parabolic_grid(s = 0.05)
vol8 = model_run(mg8, z8, s8, oid, storm_duration = 7200, model_run_time = 12000, rainfall_mmhr = 100)
print(vol8)
