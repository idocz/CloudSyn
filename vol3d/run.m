data = load('data.mat');
vol_data = data.beta;
h = vol3d('cdata',vol_data);
rotate3d on;
view 3;