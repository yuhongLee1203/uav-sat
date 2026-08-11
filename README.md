# Recurrent Visual Measurement + External Kalman v15

Plain RNN produces visual position measurement and measurement variance. External Kalman [x,y,vx,vy] performs inertia prediction, centers the full 6x6 search, then updates to final position. No waypoint/leg state, no hard forward mask, no RNN speed propagation. Maximum Kalman speed/final step is 10 m/frame and zero speed is allowed. GT is training supervision only and never enters the RNN.
