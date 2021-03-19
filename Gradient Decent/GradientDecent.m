clc
clear all
close all
warning off
syms x y
fg = x*x +3*x*y +4*y*y;
fsurf(fg,[-10 10 -10 10]);
pause(8);
hold on;
x = 10;  
y = 10;
alpha = 0.001;
for i=1:50000
    x = x - alpha * (2*x+3*y);
    y = y - alpha * (3*x+8*y); 
    fg = x*x +3*x*y +4*y*y;
    plot3(x,y,fg,'ko','linewidth',5);
    pause(0.1);
end    

