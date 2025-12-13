<div align="center">
<h1> PGTP-Net:<br>  Pose-Guided Trajectory Prediction Network </h1>
<h3>Sai Jayakar Vanam, Rithvik Shivva, Umar Hassan Makki Mohammad, Priyanka Lakariya
</h3>
<h4> <i> ME5550 Mobile Robotics, Northeastern University, Boston </i></h4>

<image src="assets/intro_overview.jpg" width="700">

</div>

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">

Understanding human motion behaviour is paramount for safe navigation of autonomous platforms. Traditional trajectory prediction relies predominantly on observed 2D position history, which inherently fails to capture subtle, anticipatory cues regarding pedestrian intent. We present PGTP-Net, a real-time multi-modal trajectory prediction system that leverages estimated 3D human pose and shape features to anticipate movement intentions 0.8-1.0 seconds earlier than position-only methods. Our dual-stream architecture combining Social-LSTM with Transformer-based pose encoding achieves 35-46% error reduction on JTA dataset with ground-truth 3D poses and 25-28% improvement on ETH/UCY with estimated 2D poses, while maintaining real-time performance suitable for autonomous navigation applications.
</br>

# Key Insight
<div align="center">

> *Human body pose reveals movement intent 0.5-1.2 seconds before that intent translates into a change in trajectory.*

</div>

Traditional methods treat pedestrians as point masses, discarding ~90% of available visual information about body configuration. PGTP-Net captures pre-movement postural adjustments—torso rotation, weight shifting, forward lean—that signal intent before trajectory changes occur.

# Getting started
Install the requirements using `pip`:
```
pip install -r requirements.txt
```
