import matplotlib.pyplot as plt

from centroidal_walk.centroidal_dyn_util import *


def simulate(f_integrator, x_start, u_vals, time_step_delta):
    steps = u_vals.shape[-1]+1
    x_curr = x_start  # start values
    x_vals = np.zeros(shape=(x_start.shape[0], steps))
    x_vals[:, 0:1] = x_curr[:]
    for i in range(x_vals.shape[1]-1):
        x_curr = f_integrator(x0=x_curr, p=u_vals[:, i])['xf']
        #print('x_curr', x_curr.shape)
        x_vals[:, i+1:i+2] = x_curr[:] # arrays are n-by-1 matrices in casadi
    t = np.arange(0, steps * time_step_delta, time_step_delta)
    return t, x_vals



def plot(t_vals, x_vals, u_vals, plot_x_vals_dict={}, plot_u_vals_dict={}):
    """
    Plots the values of x(t) and u(t) against time.

    Parameters:
    - t_vals (array-like): Time values.
    - x_vals (dims, timestep): Values of x at different time points.
    - u_vals (dims, timestep): Values of u at different time points.
    - plot_x_vals_dict (dict): Dictionary specifying the elements of x_vals to plot.
                              The keys are the names of the elements, and the values are
                              the corresponding indices of the elements in x_vals.
    - plot_u_vals_dict (dict): Dictionary specifying the elements of u_vals to plot.
                              The keys are the names of the elements, and the values are
                              the corresponding indices of the elements in u_vals.
    """
    fig, axs = plt.subplots(2)
    axs[0].set_title("$x(t)$")
    for name, el_index in plot_x_vals_dict.items():
        #axs[0].plot(t_vals, x_vals[el_index], 'tab:orange', label=name)
        axs[0].plot(t_vals, x_vals[el_index], label=name)
    axs[0].legend()
    axs[1].set_title("$u(t)$")
    for name, el_index in plot_u_vals_dict.items():
        axs[1].plot(t_vals, u_vals[el_index], label=name)
    # plt.plot(t, x_vals[1], label='dphi')
    axs[1].legend()

    plt.xlabel('time [s]')
    plt.subplots_adjust(hspace=0.5)
    #fig.legend()
    plt.draw()
    plt.pause(0.01)
    #plt.show()

def plot_multi(t_vals, plot_list):
    """
    Plots the values of x(t) and u(t) against time.

    Parameters:
    - t_vals (array-like): Time values.
    - plot_list (list):     each element corresponds to one subfigure, each element is a dict

    Example:
        plot_multi(t_vals, [
            {'title': 'A',
            'plots': {
                'lineA': arrayWithVals
            }
            }
        ]
        )
    """
    fig, axs = plt.subplots(len(plot_list))
    for sub_plt_id, sub_plt in enumerate(plot_list):
        axs[sub_plt_id].set_title(sub_plt['title'])
        if len(sub_plt['plots']) != 0:
            for name, el in sub_plt['plots'].items():
                #axs[0].plot(t_vals, x_vals[el_index], 'tab:orange', label=name)
                axs[sub_plt_id].plot(t_vals[0:el.shape[-1]], el, label=name)
        axs[sub_plt_id].legend()
    fig.set_figheight(9)
    plt.xlabel('time [s]')
    plt.subplots_adjust(hspace=0.5)
    #fig.legend()
    plt.draw()
    plt.pause(0.01)



def simulate_and_plot(f_integrator, x_start, u_vals, time_step_delta, plot_x_vals_dict={}, plot_u_vals_dict={}, show_plot=True):
    """
    Simulates a system and plots the results.

    Parameters:
    - f_integrator (function): The integration function used for simulation.
    - x_start: The initial state of the system.
    - u_vals (dims, timestep): Values of u at different time points.
    - time_step_delta: The time step size for simulation.
    - plot_x_vals_dict (dict): Dictionary specifying the elements of x_vals to plot.
                              The keys are the names of the elements, and the values are
                              the corresponding indices of the elements in x_vals.
    - plot_u_vals_dict (dict): Dictionary specifying the elements of u_vals to plot.
                              The keys are the names of the elements, and the values are
                              the corresponding indices of the elements in u_vals.
    """
    t_vals, x_vals = simulate(f_integrator, x_start, u_vals, time_step_delta)
    #print("t_vals", t_vals)
    #print("x_vals", x_vals)
    if show_plot:
        plot(t_vals, x_vals, u_vals, plot_x_vals_dict, plot_u_vals_dict)

    return t_vals, x_vals




def animate_humanoid(t_vals,
                     x_com_pos_vals,
                     x_com_rotation_vals,
                     x_foot1_pos_vals,
                     u_foot1_force_vals,
                     x_foot2_pos_vals=[],
                     u_foot2_force_vals=[],
                     foot_kin_constraint_box_center_rel: list[np.ndarray] = None,
                     foot_kin_constraint_box_size: np.ndarray = None,
                     show_movement_trails_len=30,
                     loop_duration_s=3,
                     initially_start=False
                     ):
    """
    Animates the simulated values over time.
    DEPRICATED!

    @t_vals             (timestep): Time values.
    @x_##_vals          (dims, timestep): Values of x at different time points of the corresponding entities.
    @loop_duration_s    time of one animation loop in seconds
    """
    import vedo as v
    v.settings.enable_default_keyboard_callbacks = False


    t = 0
    timesteps_len = t_vals.shape[0]
    dt_ms_per_timestep = (int) ((1000*loop_duration_s)/timesteps_len)

    # rotation values in degree
    x_com_rotation_vals__deg = rad_to_deg(x_com_rotation_vals)


    class LegObjectAbstract:
        def update(self, t):
            pass
        def reset_trails(self):
            pass
        def get_plt_objects(self):
            return []

    class LegObject(LegObjectAbstract):
        def __init__(self,
                     x_foot_pos_vals,
                     u_foot_force_vals,
                     kin_constraint_box_center_rel=None,
                     kin_constraint_box_size=None
                     ):
            self.x_foot_pos_vals = x_foot_pos_vals
            self.u_foot_force_vals = u_foot_force_vals
            self.kin_constraint_box_center_rel = kin_constraint_box_center_rel
            self.leg = v.Line([0, 0], [1, 1]).lineWidth(5)
            self.foot = v.Sphere(pos=foot1_pos_global_first, r=0.04, c="blue5", alpha=0.5)
            self.foot.add_trail(n=trail_len)
            self.foot_force = v.Arrow(head_radius=0.05, shaft_radius=0.02)
            self.foot_kin_box = None
            if kin_constraint_box_size is not None:
                self.foot_kin_box = v.Box(x_com_pos_vals[:, 0],
                                          kin_constraint_box_size[0]*2,
                                          kin_constraint_box_size[1]*2,
                                          kin_constraint_box_size[2]*2,
                                          c=[0.5, 1, 0.5],
                                          alpha=0.5
                                          ).linewidth(0.2)

        def update(self, t):
            foot_pos_global = self.x_foot_pos_vals[:, t]  # get_foot_pos_global(
            #    pos_pos_rel_to_body=x_foot_pos_vals[:, t],
            #    com_pos=x_com_pos_vals[:, 0],
            #    com_rotation=x_com_rotation_vals[::-1, 0]
            # )
            self.leg.stretch(x_com_pos_vals[:, t], foot_pos_global)
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
                rotate_object_zyx(self.foot_kin_box, x_com_rotation_vals__deg[:, t])
                #rot_matrix = R.from_euler('zxy', x_com_rotation_vals__deg[:, t], degrees=True).as_matrix()
                rot_matrix = get_rot_matrix_euler(x_com_rotation_vals__deg[:, t])
                kin_box_pos = rot_matrix @ (self.kin_constraint_box_center_rel) + x_com_pos_vals[:, t]
                self.foot_kin_box.pos(kin_box_pos)

        def reset_trails(self):
            self.foot.trail_points = [foot1_pos_global_first] * len(self.foot.trail_points)

        def get_plt_objects(self):
            if self.foot_kin_box is not None:
                return [self.leg, self.foot, self.foot_force, self.foot_kin_box]
            else:
                return [self.leg, self.foot, self.foot_force]


    foot1_pos_global_first = x_com_pos_vals[:, 0] #get_foot_pos_global(
    #    pos_pos_rel_to_body=x_foot_pos_vals[:, 0],
    #    com_pos=x_com_pos_vals[:, 0],
    #    com_rotation=x_com_rotation_vals[::-1, 0]
    #)

    #world = v.Box([0,0,0], 10, 10, 10).wireframe()
    trail_len = show_movement_trails_len
    box_size = 0.25
    box = v.Box(x_com_pos_vals[:, 0], box_size*1.2, box_size, box_size, c=[1,0.5,0.5]).linewidth(0.2)#.linecolor("black")
    box.add_trail(n=trail_len)

    leg_and_foot1 = LegObject(x_foot1_pos_vals,
                              u_foot1_force_vals,
                              None if foot_kin_constraint_box_center_rel is None else foot_kin_constraint_box_center_rel[0],
                              foot_kin_constraint_box_size
                              )
    leg_and_foot2 = LegObjectAbstract()
    if len(x_foot2_pos_vals) != 0 and len(u_foot2_force_vals) != 0:
        leg_and_foot2 = LegObject(x_foot2_pos_vals,
                                  u_foot2_force_vals,
                                  None if foot_kin_constraint_box_center_rel is None else foot_kin_constraint_box_center_rel[1],
                                  foot_kin_constraint_box_size
                                  )

    text = v.Text2D(pos="bottom-left", font="FiraMonoMedium", s=0.7)


    def animation_frame(event, increase_t=True, rest=False):
        nonlocal t
        #i = event.time
        #box.pos(np.sin(timestep * 0.01), 0, 0).rotate_x(sin(timestep*0.01))

        #box.pos(x_com_pos_vals[0, t], x_com_pos_vals[1, t], x_com_pos_vals[2, t])

        #box.SetOrientation(x_com_rotation_vals__deg[0, t], x_com_rotation_vals__deg[1, t], x_com_rotation_vals__deg[2, t])
        #box.orientation()
        rotate_object_zyx(box, x_com_rotation_vals__deg[:, t])
        #box.SetOrientation(0, 0, 0)
        #box.rotate_y(x_com_rotation_vals[1, t])
        #box.rotate_z(x_com_rotation_vals[2, t])
        box.pos(x_com_pos_vals[:, t])
        box.update_trail()

        leg_and_foot1.update(t)
        leg_and_foot2.update(t)


        text.text(f"Time {round(t_vals[t], ndigits=1)}s \t \tTimestep {t} / {timesteps_len-1}")
        plt.render()

        if increase_t:
            t += 1
        if t >= timesteps_len:
            t = 0
        if t >= timesteps_len or rest:
            # reset trails
            box.trail_points = [x_com_pos_vals[:, t]] * len(box.trail_points)
            #foot1.trail_points = [foot1_pos_global_first] * len(foot1.trail_points)
            leg_and_foot1.reset_trails()
            leg_and_foot2.reset_trails()
            #box.update_trail()

    plt = v.Plotter(interactive=True)
    plt += box
    plt += leg_and_foot1.get_plt_objects()
    plt += leg_and_foot2.get_plt_objects()
    #plt += [foot1_force_x]
    plt += text
    plt += v.Axes(xrange=(-1.5,1.5), yrange=(-1.5, 1.5), zrange=(0,1.5))
    animation_frame(None)

    # start stop button
    def button_start_stop_callback():
        nonlocal timer_id, button_stop, t
        if "Play" in button_stop.status():
            # timestep = 0
            # instruct to call handle_timer() every dt_ms_per_timestep msec:
            timer_id = plt.timer_callback("start", dt=dt_ms_per_timestep)
        else:
            plt.timer_callback("stop", timer_id)
        button_stop.switch()
    button_stop = plt.add_button(button_start_stop_callback, states=[" Play ", "Pause"], pos=(0.05, 0.95), size=22)

    def key_pressed_callback(event):
        nonlocal t
        changed = False
        if event.keyPressed == "space":
            button_start_stop_callback()
            plt.render()
        if event.keyPressed == "Left":
            t -= 1
            changed = True
        if event.keyPressed == "Right":
            t += 1
            changed = True
        if t >= timesteps_len:
            t = 0
            animation_frame(None, increase_t=False, rest=True)
        if t < 0:
            t = timesteps_len-1
            animation_frame(None, increase_t=False, rest=True)
        if changed:
            animation_frame(None, increase_t=False)
    plt.add_callback("Char", key_pressed_callback)

    # create timer with callback
    plt.add_callback("timer", animation_frame)
    if initially_start:
        timer_id = plt.timer_callback("start", dt=dt_ms_per_timestep)
        button_stop.status("Pause")

    plt.show(viewup='z')


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
