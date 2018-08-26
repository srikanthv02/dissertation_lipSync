
target = testVideo;
target2 = testPrediction;
figure()
for i = 1:length(target)
   
    clf
    
   x_pos = target(i,1:2:end);
   y_pos = target(i,2:2:end);
   
   x_pos2 = target2(i,1:2:end);
   y_pos2 = target2(i,2:2:end);
   
   y_pos2(15:31) = y_pos2(15:31) + randn(1,17)*2;
%     y_pos2 = y_pos2 + randn(1,36)*1;
   
   scatter(x_pos,-y_pos)
   hold on
   scatter(x_pos2, -y_pos2, 'r');
   xlim([500,1500]);
   ylim([-1100, -550]);
   pause();

end