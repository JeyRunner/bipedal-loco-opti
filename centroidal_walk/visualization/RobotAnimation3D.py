import matplotlib.pyplot as plt
import numpy as np
from vedo import interactor_modes

from centroidal_walk.centroidal_dyn_util import *
import vedo as v



class RobotAnimation3D:

    class LegObjectAbstract:
        def __init__(self):
            self.foot_pos_global_first = np.zeros((3))
            self.x_com_pos_vals = np.zeros((3, 1))
            self.x_com_rotation_vals__deg = np.zeros((3, 1))
            self.x_foot_pos_vals = np.zeros((3, 1))
            self.u_foot_force_vals = np.zeros((3, 1))

        def set_trajectory_values(self,
                                  foot_pos_global_first,
                                  x_com_pos_vals,
                                  x_com_rotation_vals__deg,
                                  x_foot_pos_vals,
                                  u_foot_force_vals
                                  ):
            self.foot_pos_global_first = foot_pos_global_first
            self.x_com_pos_vals = x_com_pos_vals
            self.x_com_rotation_vals__deg = x_com_rotation_vals__deg
            self.x_foot_pos_vals = x_foot_pos_vals
            self.u_foot_force_vals = u_foot_force_vals

        def update(self, t):
            pass

        def reset_trails(self):
            pass

        def get_plt_objects(self):
            return []

    class LegObject(LegObjectAbstract):
        def __init__(self,
                     kin_constraint_box_center_rel=None,
                     kin_constraint_box_size=None,
                     trail_len=30
                     ):
            RobotAnimation3D.LegObjectAbstract.__init__(self)

            # dummy values
            self.foot_pos_global_first = np.zeros(3)
            self.x_com_pos_vals = np.zeros((3, 1))


            self.kin_constraint_box_center_rel = kin_constraint_box_center_rel
            self.leg = v.Line([0, 0], [1, 1]).lineWidth(5)
            self.foot = v.Sphere(pos=self.foot_pos_global_first, r=0.04, c="blue5", alpha=0.5)
            self.foot.add_trail(n=trail_len)
            self.foot_force = v.Arrow(head_radius=0.05, shaft_radius=0.02)
            self.foot_kin_box = None
            if kin_constraint_box_size is not None:
                self.foot_kin_box = v.Box(self.x_com_pos_vals[:, 0],
                                          kin_constraint_box_size[0] * 2,
                                          kin_constraint_box_size[1] * 2,
                                          kin_constraint_box_size[2] * 2,
                                          c=[0.5, 1, 0.5],
                                          alpha=0.5
                                          ).linewidth(0.2)

        def update(self, t):
            foot_pos_global = self.x_foot_pos_vals[:, t]  # get_foot_pos_global(
            #    pos_pos_rel_to_body=x_foot_pos_vals[:, t],
            #    com_pos=x_com_pos_vals[:, 0],
            #    com_rotation=x_com_rotation_vals[::-1, 0]
            # )
            self.leg.stretch(self.x_com_pos_vals[:, t], foot_pos_global)
            self.foot.pos(foot_pos_global)
            self.foot.update_trail()

            foot_force_val = (self.u_foot_force_vals[:, t] if t < self.u_foot_force_vals.shape[1] else np.zeros(3))
            foot_force_start_pos = (foot_pos_global - foot_force_val * 0.01)
            if np.abs(foot_force_val).max() <= 0.0001:
                self.foot_force.opacity(0)
            else:
                self.foot_force.stretch(foot_force_start_pos, foot_pos_global)
                self.foot_force.opacity(1)

            # kin box
            if self.kin_constraint_box_center_rel is not None:
                rotate_object_zyx(self.foot_kin_box, self.x_com_rotation_vals__deg[:, t])
                # rot_matrix = R.from_euler('zxy', x_com_rotation_vals__deg[:, t], degrees=True).as_matrix()
                rot_matrix = get_rot_matrix_euler(self.x_com_rotation_vals__deg[:, t])
                kin_box_pos = rot_matrix @ (self.kin_constraint_box_center_rel) + self.x_com_pos_vals[:, t]
                self.foot_kin_box.pos(kin_box_pos)

        def reset_trails(self):
            self.foot.trail_points = [self.foot_pos_global_first] * len(self.foot.trail_points)

        def get_plt_objects(self):
            if self.foot_kin_box is not None:
                return [self.leg, self.foot, self.foot_force, self.foot_kin_box]
            else:
                return [self.leg, self.foot, self.foot_force]


    def set_target_pos_changed_callback(self, callback):
        self.target_pos_changed_callback = callback


    def __init__(self,
                 foot_kin_constraint_box_center_rel: list[np.ndarray] = None,
                 foot_kin_constraint_box_size: np.ndarray = None,
                 num_feet=2,
                 show_movement_trails_len=30,
                 ):
        v.settings.enable_default_keyboard_callbacks = False

        self.foot_kin_constraint_box_center_rel = foot_kin_constraint_box_center_rel
        self.foot_kin_constraint_box_size = foot_kin_constraint_box_size
        self.show_movement_trails_len = show_movement_trails_len
        self.num_feet = num_feet
        self.target_pos_changed_callback = None
        self.showing = False
        self.once = False
        trail_len = show_movement_trails_len

        # default empty params
        self.t_vals = np.zeros((3, 1))
        self.t = 0
        self.timesteps_len = 1
        self.dt_ms_per_timestep = 1
        self.x_com_pos_vals = np.zeros((3, 1))
        self.x_com_rotation_vals = np.zeros((3, 1))
        self.x_com_rotation_vals__deg = np.zeros((3, 1))
        self.x_feet_pos_vals = [np.zeros((3, 1))]
        self.u_feet_force_vals = [np.zeros((3, 1))]
        self.x_foot2_pos_vals = np.zeros((3, 1))
        self.u_foot2_force_vals = np.zeros((3, 1))
        self.loop_duration_s = 3,
        self.timer_id = -1


        # create shapes
        self.ground = v.Plane(s=[3, 3]).wireframe(False).opacity(0.0001)
        box_size = 0.25# .linecolor("black")
        self.box = (v.Box(np.zeros(3), box_size * 1.2, box_size, box_size, c=[1, 0.5, 0.5])
                    .pickable(False).draggable(False)
                    .linewidth(0.2))
        self.box.add_trail(n=trail_len)

        self.leg_and_foot_list = []
        for foot_i in range(self.num_feet):
            leg_and_foot = RobotAnimation3D.LegObject(
                                      None if foot_kin_constraint_box_center_rel is None else
                                        foot_kin_constraint_box_center_rel[foot_i],
                                      foot_kin_constraint_box_size,
                                      trail_len=trail_len
                                      )
            self.leg_and_foot_list.append(leg_and_foot)

        self.text = v.Text2D(pos="bottom-left", font="FiraMonoMedium", s=0.7)
        self.com_target_marker = v.Sphere(pos=np.zeros(3), r=0.02, c="red5", alpha=0.9).pickable(True).draggable(True)

        # add shapes
        #v.settings.enable_default_mouse_callbacks = False
        #mode = interactor_modes.BlenderStyle()
        self.plt = v.Plotter(interactive=True)#.user_mode(mode)
        self.plt += self.box
        for leg_and_foot in self.leg_and_foot_list:
            self.plt += leg_and_foot.get_plt_objects()
        #plt += [foot1_force_x]
        self.plt += self.text
        self.plt += self.com_target_marker
        self.plt += v.Axes(xrange=(-1.5,1.5), yrange=(-1.5, 1.5), zrange=(0,1.5))
        self.plt += self.ground



        # start stop button
        def button_start_stop_callback():
            if "Play" in self.button_stop.status():
                # timestep = 0
                # instruct to call handle_timer() every dt_ms_per_timestep msec:
                self.timer_id = self.plt.timer_callback("start", dt=self.dt_ms_per_timestep)
            else:
                self.timer_id = self.plt.timer_callback("stop", self.timer_id)
            self.button_stop.switch()
            self.plt.render()

        self.button_stop = self.plt.add_button(button_start_stop_callback, states=[" Play ", "Pause"], pos=(0.05, 0.95),
                                               size=22)
        # keyboard
        def key_pressed_callback(event):
            changed = False
            if event.keyPressed == "space":
                button_start_stop_callback()
                self.plt.render()
            if event.keyPressed == "Left":
                self.t -= 1
                changed = True
            if event.keyPressed == "Right":
                self.t += 1
                changed = True
            if self.t >= self.timesteps_len:
                self.t = 0
                self.__animation_frame(None, increase_t=False, rest=True)
            if self.t < 0:
                self.t = self.timesteps_len - 1
                self.__animation_frame(None, increase_t=False, rest=True)
            if changed:
                self.__animation_frame(None, increase_t=False)

            if event.keyPressed == "Return" and not self.com_target_marker_fixed:
                if self.target_pos_changed_callback is not None:
                    self.com_target_marker_fixed = True
                    self.target_pos_changed_callback(self.com_target_marker_pos_2d)

        self.plt.add_callback("Char", key_pressed_callback)

        # create timer with callback
        self.plt.add_callback("timer", lambda event: self.__animation_frame(event))

        # drag target pos maker callback
        self.com_target_marker_fixed = False
        self.com_target_marker_pos_2d = None
        def mouse_callback(event):
            if event.picked3d is None:
                return
            if self.target_pos_changed_callback is not None:
                if not self.com_target_marker_fixed:
                    point = self.ground.closest_point(event.picked3d)
                    #print(event)
                    #print(point)
                    self.com_target_marker.pos(point)
                    self.com_target_marker_pos_2d = point[0:2]
                    self.plt.render()

            #print(event)
        self.plt.add_callback('mouse hovering', mouse_callback)

        # set initial
        #self.__animation_frame(None)
        #self.plt.show(viewup='z')




    def __animation_frame(self, event, increase_t=True, rest=False):
        #i = event.time
        #box.pos(np.sin(timestep * 0.01), 0, 0).rotate_x(sin(timestep*0.01))

        #box.pos(x_com_pos_vals[0, t], x_com_pos_vals[1, t], x_com_pos_vals[2, t])

        #box.SetOrientation(x_com_rotation_vals__deg[0, t], x_com_rotation_vals__deg[1, t], x_com_rotation_vals__deg[2, t])
        #box.orientation()
        rotate_object_zyx(self.box, self.x_com_rotation_vals__deg[:, self.t])
        #box.SetOrientation(0, 0, 0)
        #box.rotate_y(x_com_rotation_vals[1, t])
        #box.rotate_z(x_com_rotation_vals[2, t])
        self.box.pos(self.x_com_pos_vals[:, self.t])
        self.box.update_trail()

        for leg_and_foot in self.leg_and_foot_list:
            leg_and_foot.update(self.t)


        self.text.text(f"Time {np.round(self.t_vals[self.t], decimals=1)}s \t \tTimestep {self.t} / {self.timesteps_len-1}")
        self.plt.render()

        if increase_t:
            self.t += 1
        if self.t >= self.timesteps_len and self.once:
            self.stop_animation_timer()

        if self.t >= self.timesteps_len or rest:
            # reset trails
            self.box.trail_points = [self.x_com_pos_vals[:, self.timesteps_len-1]] * len(self.box.trail_points)
            #foot1.trail_points = [foot1_pos_global_first] * len(foot1.trail_points)
            for leg_and_foot in self.leg_and_foot_list:
                leg_and_foot.reset_trails()
            #box.update_trail()

        if self.t >= self.timesteps_len:
            self.t = 0



    def stop_animation_timer(self):
        self.button_stop.status(" Play ")
        self.plt.render()
        self.plt.timer_callback("stop", self.timer_id)






    def animate_humanoid(self,
                         t_vals,
                         x_com_pos_vals,
                         x_com_rotation_vals,
                         x_feet_pos_vals: list[np.ndarray],
                         u_feet_force_vals : list[np.ndarray],
                         loop_duration_s=3,
                         initially_start=False,
                         once=True,
                         show_new_window=False
                         ):
        """
        Animates the simulated values over time.

        :param t_vals:             (timestep): Time values.
        :param x_##_vals:          list of elements (dims, timestep) - Values of x at different time points of the corresponding entities.
                                    Each list item corresponds to one foot.
        @loop_duration_s    time of one animation loop in seconds
        """
        self.t_vals = t_vals
        self.x_com_pos_vals = x_com_pos_vals
        self.x_com_rotation_vals = x_com_rotation_vals
        self.x_feet_pos_vals = x_feet_pos_vals
        self.x_feet_force_vals = u_feet_force_vals
        self.loop_duration_s = loop_duration_s
        self.once = once

        # rotation values in degree
        self.x_com_rotation_vals__deg = rad_to_deg(x_com_rotation_vals)

        for foot_i, leg_and_foot in enumerate(self.leg_and_foot_list):
            leg_and_foot.set_trajectory_values(
                                      x_feet_pos_vals[foot_i][:, 0],
                                      x_com_pos_vals,
                                      self.x_com_rotation_vals__deg,
                                      x_feet_pos_vals[foot_i],
                                      u_feet_force_vals[foot_i]
                                      )


        self.t = 0
        self.timesteps_len = t_vals.shape[0]
        self.dt_ms_per_timestep = (int) ((1000*loop_duration_s)/self.timesteps_len)



        self.__animation_frame(None, increase_t=False, rest=True)


        foot1_pos_global_first = x_com_pos_vals[:, 0] #get_foot_pos_global(
        #    pos_pos_rel_to_body=x_foot_pos_vals[:, 0],
        #    com_pos=x_com_pos_vals[:, 0],
        #    com_rotation=x_com_rotation_vals[::-1, 0]
        #)

        if initially_start:
            self.timer_id = self.plt.timer_callback("stop", self.timer_id) # stop old timer
            self.timer_id = self.plt.timer_callback("start", dt=self.dt_ms_per_timestep)
            self.button_stop.status("Pause")
            self.plt.render()

        if not self.showing or show_new_window:
            self.plt.show(viewup='z')
            self.showing = True










def rotate_object_zyx(o, rotation_xyz):
    import vtk
    T = vtk.vtkTransform()
    T.RotateZ(rotation_xyz[2])
    T.RotateY(rotation_xyz[1])
    T.RotateX(rotation_xyz[0])
    o.SetOrientation(T.GetOrientation())


def get_rot_matrix_euler(rotation_xyz):
    import vtk
    T = vtk.vtkTransform()
    T.RotateZ(rotation_xyz[2])
    T.RotateY(rotation_xyz[1])
    T.RotateX(rotation_xyz[0])
    R = np.zeros((3,3))
    # to np array
    for i in range(3):
        for j in range(3):
            R[i, j] = T.GetMatrix().GetElement(i, j)
    return R
