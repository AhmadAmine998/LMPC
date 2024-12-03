import signal
import numpy as np
from FTOCP import FTOCP
from LMPC import LMPC
import os
import datetime
import copy
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
import control as ct
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

curr_pos = []
curr_opt_traj = None

class Plottings:
    def __init__(self, feasible_traj, rec, save_dir, enable=True):
        self.enable = enable
        self.feasible_traj = feasible_traj
        x_obs = []
        y_obs = []
        for i in np.linspace(0,2*np.pi,1000):
            x_obs.append(30 + 10*np.cos(i))
            y_obs.append(0 + 10*np.sin(i))
        self.obs = np.array([x_obs, y_obs]).T
        self.rec = rec
        self.save_dir = save_dir
        if not self.enable and not enable: return
        
    def show(self):
        plt.show()
    
    def show_pause(self):
        plt.draw()
        while plt.waitforbuttonpress(0.2) is None:
            if self.win_closed:
                break
        plt.close(self.fig)
        self.win_closed = False
    
    def save_fig(self, filename):
        plt.savefig(filename, bbox_inches='tight')
        
    def close_all(self):
        plt.close('all')

    def grid(self, ax_num):
        self.axs[ax_num].grid(which='both', axis='both')

    def get_fig(self, grid=[1, 1], figsize=[8, 6], dpi=100, gridline=False):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        self.fig = fig
        self.fig.tight_layout()
        self.gs = GridSpec(grid[0], grid[1])
        self.axs = []
        for ind in range(np.prod(grid)):
            self.axs.append(fig.add_subplot(self.gs[ind]))
        if gridline:
            for ind in range(len(self.axs)):
                self.grid(ind)
            
        def handle_close(evt):
            self.win_closed = True
        self.fig.canvas.mpl_connect('close_event', handle_close)
        return self.axs
    
    def background(self, color, ax_num):
        self.axs[ax_num].set_facecolor(color)
    
    def colorbar(self, ax_num, cmap='viridis', data_ind=0, location='right', label=''):
        if len(self.axs[ax_num].collections) > 0:
            cbar = plt.colorbar(self.axs[ax_num].collections[data_ind], 
                                     ax=self.axs[ax_num], cmap=cmap, 
                                     location=location, label=label)
        elif len(self.axs[ax_num].images) > 0:
            cbar = plt.colorbar(self.axs[ax_num].images[data_ind], 
                                     ax=self.axs[ax_num], cmap=cmap, 
                                     location=location, label=label)
        
    def plot_trajectory(self, key_option, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.get_fig([1, 1])
        axs[0].scatter(self.feasible_traj[:, 1], self.feasible_traj[:, 2], s=1, c='grey')
        axs[0].scatter(self.obs[:, 0], self.obs[:, 1], s=1, c='k')
        axs[0].scatter(self.rec.xy_lap_record[-1][:, 0], self.rec.xy_lap_record[-1][:, 1], s=1, c='r')
        axs[0].set_title(f'Trajectory {laptime:.2f}s')
        self.save_fig(self.save_dir + 'trajectory')
        if key_option == '2': self.show()
        self.close_all()
        
    def plot_trajectory_history(self, lap_cnt, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.get_fig([1, 1])
        axs[0].scatter(self.feasible_traj[:, 1], self.feasible_traj[:, 2], s=1, c='grey')
        axs[0].scatter(self.obs[:, 0], self.obs[:, 1], s=1, c='k')
        axs[0].scatter(self.rec.xy_lap_record[-1][:, 0], self.rec.xy_lap_record[-1][:, 1], s=1, c='r')
        axs[0].set_title(f'Trajectory {laptime:.2f}s')
        if not os.path.exists(self.save_dir + 'traj_evo/'):
            os.makedirs(self.save_dir + 'traj_evo/')
        self.save_fig(self.save_dir + 'traj_evo/' + 'trajectory_' + str(lap_cnt))
        self.close_all()
        
    def plot_laptime_speed(self, key_option, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.get_fig([2, 1], gridline=True)
        axs[0].plot(np.arange(len(self.rec.laptime_record)), self.rec.laptime_record)
        axs[0].set_title(f'Laptime vs. Iteration {laptime:.2f}s')
        axs[1].plot(np.arange(len(self.rec.mean_vels)), self.rec.mean_vels)
        axs[1].set_title(f'Average speed vs. Iteration {laptime:.2f}s')
        self.save_fig(self.save_dir + 'laptime_speed')
        if key_option == '2':
            self.show()
        self.close_all()
            
    def plot_ss_violation(self, key_option, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.get_fig([2, 1], gridline=True)
        axs[0].plot(np.arange(len(self.rec.ss_max_record)), self.rec.ss_max_record)
        axs[0].set_title(f'Max ss_violation vs. Iteration {laptime:.2f}s')
        axs[1].plot(np.arange(len(self.rec.ss_average_record)), self.rec.ss_average_record)
        axs[1].set_title(f'Average ss_violation vs. Iteration {laptime:.2f}s')
        self.save_fig(self.save_dir + 'ss_violation')
        if key_option == '2':
            self.show()
        self.close_all()

    def plot_boundary_violation(self, key_option, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.get_fig([2, 1], gridline=True)
        axs[0].plot(np.arange(len(self.rec.boundary_max_record)), self.rec.boundary_max_record)
        axs[0].set_title(f'Max boundary violation vs. Iteration {laptime:.2f}s')
        axs[1].plot(np.arange(len(self.rec.boundary_average_record)), self.rec.boundary_average_record)
        axs[1].set_title(f'Average boundary violation vs. Iteration {laptime:.2f}s')
        self.save_fig(self.save_dir + 'boundary_violation')
        if key_option == '2':
            self.show()
        self.close_all()
    
    def plot_speed_position(self, key_option, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.get_fig([1, 1])
        self.background((0, 0, 0), 0)
        axs[0].scatter(self.rec.xy_lap_record[-1][:, 0], self.rec.xy_lap_record[-1][:, 1], s=2, c=self.rec.xy_lap_record[-1][:, 3], cmap='coolwarm')
        axs[0].set_title(f'Speed vs. Position {laptime:.2f}s')
        self.colorbar(0, cmap='coolwarm')
        axs[0].scatter(self.obs[:, 0], self.obs[:, 1], s=1, c='k')
        self.save_fig(self.save_dir + 'speed_position')
        if key_option == '2':
            self.show()
        self.close_all()
        
    def plot_min_speed_position(self, key_option, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.get_fig([1, 1])
        self.background((0, 0, 0), 0)
        axs[0].scatter(self.rec.min_states[:, 0], self.rec.min_states[:, 1], s=2, c=self.rec.min_states[:, 3], cmap='coolwarm')
        axs[0].set_title(f'Speed vs. Position {laptime:.2f}s')
        self.colorbar(0, cmap='coolwarm')
        axs[0].scatter(self.obs[:, 0], self.obs[:, 1], s=1, c='k')
        self.save_fig(self.save_dir + 'speed_position')
        if key_option == '2':
            self.show()
        self.close_all()


    def plot_speed_position_history(self, lap_cnt, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.get_fig([1, 1])
        self.background((0, 0, 0), 0)
        axs[0].scatter(self.rec.xy_lap_record[-1][:, 0], self.rec.xy_lap_record[-1][:, 1], s=2, c=self.rec.xy_lap_record[-1][:, 3], cmap='coolwarm')
        axs[0].set_title(f'Speed vs. Position {laptime:.2f}s')
        self.colorbar(0, cmap='coolwarm')
        axs[0].scatter(self.obs[:, 0], self.obs[:, 1], s=1, c='k')
        if not os.path.exists(self.save_dir + 'speed_evo/'):
            os.makedirs(self.save_dir + 'speed_evo/')
        self.save_fig(self.save_dir + 'speed_evo/' + 'speed_position_' + str(lap_cnt))
        self.close_all()
        
    def plot_acceleration_position(self, key_option, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.get_fig([1, 1])
        self.background((0, 0, 0), 0)
        axs[0].scatter(self.rec.xy_lap_record[-1][:, 0], self.rec.xy_lap_record[-1][:, 1], s=2, c=self.rec.control_records[-1][:, 0], cmap='coolwarm')
        axs[0].set_title(f'Acceleration vs. Position {laptime:.2f}s')
        self.colorbar(0, cmap='coolwarm')
        axs[0].scatter(self.obs[:, 0], self.obs[:, 1], s=1, c='k')
        self.save_fig(self.save_dir + 'acceleration_position')
        if key_option == '2':
            self.show()
        self.close_all()
        
    def plot_acceleration_position_history(self, lap_cnt, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.get_fig([1, 1])
        self.background((0, 0, 0), 0)
        axs[0].scatter(self.rec.xy_lap_record[-1][:, 0], self.rec.xy_lap_record[-1][:, 1], s=2, c=self.rec.control_records[-1][:, 0], cmap='coolwarm')
        axs[0].set_title(f'Acceleration vs. Position {laptime:.2f}s')
        self.colorbar(0, cmap='coolwarm')
        axs[0].scatter(self.obs[:, 0], self.obs[:, 1], s=1, c='k')
        if not os.path.exists(self.save_dir + 'accel_evo/'):
            os.makedirs(self.save_dir + 'accel_evo/')
        self.save_fig(self.save_dir + 'accel_evo/' + 'acceleration_position_' + str(lap_cnt))
        self.close_all()
        
    def plot_ss_lamb_history(self, lap_cnt, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.get_fig([1, 1])
        self.background((0, 0, 0), 0)
        axs[0].scatter(self.rec.xy_lap_record[-1][:, 0], self.rec.xy_lap_record[-1][:, 1], s=2, 
                       c=np.array(self.rec.lamb_record)[:, 0], cmap='coolwarm')
        axs[0].set_title(f'Safeset Lambs vs. Position {laptime:.2f}s')
        self.colorbar(0, cmap='coolwarm')
        axs[0].scatter(self.obs[:, 0], self.obs[:, 1], s=1, c='k')
        if not os.path.exists(self.save_dir + 'sslamb_evo/'):
            os.makedirs(self.save_dir + 'sslamb_evo/')
        self.save_fig(self.save_dir + 'sslamb_evo/' + 'ss_lamb_position_' + str(lap_cnt))
        self.close_all()
        
    def plot_boundary_lamb_history(self, lap_cnt, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.get_fig([1, 1])
        self.background((0, 0, 0), 0)
        axs[0].scatter(self.rec.xy_lap_record[-1][:, 0], self.rec.xy_lap_record[-1][:, 1], s=2, 
                       c=np.array(self.rec.lamb_record)[:, 1], cmap='coolwarm')
        axs[0].set_title(f'Boundary Lambs vs. Position {laptime:.2f}s')
        self.colorbar(0, cmap='coolwarm')
        axs[0].scatter(self.obs[:, 0], self.obs[:, 1], s=1, c='k')
        if not os.path.exists(self.save_dir + 'boundarylamb_evo/'):
            os.makedirs(self.save_dir + 'boundarylamb_evo/')
        self.save_fig(self.save_dir + 'boundarylamb_evo/' + 'boundary_lamb_position_' + str(lap_cnt))
        self.close_all()
        
    def plot_lap_lambs(self, lap_cnt, laptime, enable=False):
        if not self.enable and not enable: return
        axs = self.get_fig([2, 1], gridline=True)
        axs[0].plot(np.arange(len(self.rec.laptime_record)), np.array(self.rec.mean_lambs)[:, 0])
        axs[0].set_title(f'Average Safeset Lambs {laptime:.2f}s')
        axs[1].plot(np.arange(len(self.rec.laptime_record)), np.array(self.rec.mean_lambs)[:, 1])
        axs[1].set_title(f'Average Boundary Lambs {laptime:.2f}s')
        self.save_fig(self.save_dir + 'lap_lambs')
        self.close_all()

def end_action(rec, save_dir, plottings):
    plottings.plot_laptime_speed(0, np.min(rec.laptime_record), enable=True)
    plottings.plot_min_speed_position(0, rec.min_laptime, enable=True)
    ## recording save
    rec.save_onefile('laptimes', # record per action laptime
                    'xy_states', # record in-loop cartesian poses
                    'xy_lap_record', # record xy_states per lap
                    'controls_lap_record', # record control per lap
                    'safe_set_xy', 
                    'control_record', 'control_records',
                    'a_opt_record', 'a_opt_records', 'mean_vels', 'traj_opts',
                    'ss_violation_record', 'ss_average_record', 'ss_max_record', 
                    'boundary_violation_record', 'boundary_average_record', 'boundary_max_record', 
                    'laptime_record',
                    'min_laptime', 'min_states', 'time_record', save_dir=save_dir)
    return

    
class Logger:
    def __init__(self, save_dir, experiment_name, create_file=True) -> None:
        from io import StringIO
        self.s = StringIO()
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        if create_file:
            self.create_file(experiment_name)
    
    def create_file(self, experiment_name):
        import datetime
        print(self.experiment_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.save_dir + experiment_name + '.txt', "a") as tgt:
            tgt.writelines('\n' + '-' * 80 + '\n')
            tgt.writelines(experiment_name + ' ' + str(datetime.datetime.now()) + '\n')
            
    def write_file(self, file):
        with open(file, "r") as src:
            with open(self.save_dir + self.experiment_name + '.py', "w") as tgt:
                tgt.write(src.read())
    
    def line(self, *line, print_line=True):
        if print_line: print(*line)
        print(*line, file = self.s)
        with open(self.save_dir + self.experiment_name + '.txt', "a") as tgt:
            tgt.writelines(self.s.getvalue())
        self.s.truncate(0)
        self.s.seek(0)
            
class ListDict:
    def __init__(self) -> None:
        pass

    def init(self, *keys):
        for key in keys:
            setattr(self, key, []) 
    
    def get_keys(self):
        return list(vars(self).keys())
    
    def list(self):
        print(self.get_keys())
            
    def pop(self, *keys, index=0):
        if len(keys) == 0:
            keys = self.get_keys()
        for key in keys:
            # print(key)
            getattr(self, key).pop(index)
    
    def save(self, *keys, save_dir=''):
        for key in keys:
            np.savez(save_dir + key, *getattr(self, key))
            
    def load(self, *keys, save_dir=''):
        for key in keys:
            setattr(self, key, list(np.load(save_dir + key + '.npz', allow_pickle=True).values()))
    
    def load_onefile_old(self, save_dir='', filename = 'data_record'):
        d = np.load(save_dir + filename + '.npz', allow_pickle=True)['arr_0'][()]
        for key in list(d.keys()):
            if hasattr(d[key], "__len__"):
                setattr(self, key, list(d[key]))
            else:
                setattr(self, key, d[key])
                
    def save_onefile(self, *keys, save_dir='', filename = 'data_record', compress=False):
        if len(keys) == 0:
            keys = self.get_keys()
        d = {}
        for key in keys:
            d[key] = {key: getattr(self, key)}
        if compress:
            np.savez_compressed(save_dir + filename, **d)
        else:
            np.savez(save_dir + filename, **d)
            
    def load_onefile(self, *keys, save_dir='', filename = 'data_record'):
        d = np.load(save_dir + filename + '.npz', allow_pickle=True)
        if len(keys) == 0:
            keys = list(d.keys())
        for key in keys:
            if hasattr(d[key][()][key], "__len__"):
                setattr(self, key, list(d[key][()][key]))
            else:
                setattr(self, key, d[key][()][key])

class LMPCTask(QtCore.QThread):
    """Thread class for running LMPC task."""
    update_signal = QtCore.pyqtSignal()  # Signal to trigger UI updates

    def __init__(self, xclFeasible, uclFeasible, ftocp, x0, itMax, safeSetOption, N=10, plot=False, parent=None):
        super().__init__(parent)
        self.xclFeasible = xclFeasible
        self.uclFeasible = uclFeasible
        self.ftocp = ftocp
        self.x0 = x0
        self.itMax = itMax
        self.safeSetOption = safeSetOption
        self.plot = plot
        self.verbose = True
        self.N = N

        self.rec = ListDict()
        self.rec.init('laptimes', # record per action laptime
                    'xy_states', # record in-loop cartesian poses
                    'xy_lap_record', # record xy_states per lap
                    'controls_lap_record', # record control per lap
                    'safe_set_xy', 
                    'control_record', 'control_records',
                    'a_opt_record', 'a_opt_records', 'mean_vels', 'traj_opts',
                    'ss_violation_record', 'ss_average_record', 'ss_max_record', 
                    'boundary_violation_record', 'boundary_average_record', 'boundary_max_record', 
                    'laptime_record',
                    'min_laptime', 'min_states', 'time_record')   
        self.rec.min_laptime = np.inf

        self.l = 20
        self.P = 24
        self.exp_name = f'lmpc_P{self.P}_l{self.l}_N{self.ftocp.N}_safeSetOption{self.safeSetOption}'
        self.save_dir = 'storedData/' + self.exp_name + '/'
        self.log = Logger(self.save_dir, self.exp_name, create_file=True)
        
    def run(self):
        itCounter = 1
        lmpc = LMPC(self.ftocp, l=self.l, P=self.P, safeSetOption=self.safeSetOption)
        lmpc.addTrajectory(self.xclFeasible, self.uclFeasible)
        plottings = Plottings(self.xclFeasible, self.rec, self.save_dir, enable=True)

        global curr_pos
        global curr_opt_traj
        meanTimeCostLMPC = []
        for itCounter in tqdm(range(1, self.itMax + 1), desc="LMPC Iterations"):
            timeStep = 0
            laptime = 0.0
            xcl = [self.x0]
            ucl = []
            timeLMPC = []
            curr_pos = []

            while np.linalg.norm(xcl[-1][:2] - self.xclFeasible[:2, -1]) >= 0.1:
                xt = xcl[timeStep]
                startTimer = datetime.datetime.now()
                lmpc.solveConvexHull(xt, verbose=0)
                deltaTimer = datetime.datetime.now() - startTimer
                timeLMPC.append(deltaTimer.total_seconds())
                # if self.plot:
                #     input("Press Enter to continue...")

                ut = lmpc.ut
                ucl.append(copy.copy(ut))
                xcl.append(copy.copy(self.ftocp.f(xcl[timeStep], ut)))
                if self.plot:
                    curr_pos.append(xcl[-1][:2])
                    curr_opt_traj = lmpc.xOpenLoop.copy()
                    self.update_signal.emit()  # Notify UI to update

                timeStep += 1
                laptime += self.ftocp.dt

                self.rec.ss_violation_record.append(lmpc.slack.T)
                obsSlack = None
                if lmpc.slackObs.T.shape == (1, self.N):
                    obsSlack = lmpc.slackObs.T
                else:
                    # Pad boundary slack with zeros
                    obsSlack = np.zeros((1, self.N))
                    obsSlack[0, :lmpc.slackObs.T.shape[1]] = lmpc.slackObs.T
                self.rec.boundary_violation_record.append(obsSlack)

                self.rec.laptimes.append(laptime)
                self.rec.xy_states.append(xcl[-1])
                self.rec.control_record.append(ut)
                self.rec.a_opt_record.append(lmpc.uOpenLoop)

            self.rec.ss_violation_record.append(lmpc.slack.T)
            if lmpc.slackObs.T.shape == (1, self.N):
                obsSlack = lmpc.slackObs.T
            else:
                # Pad boundary slack with zeros
                obsSlack = np.zeros((1, self.N))
                obsSlack[0, :lmpc.slackObs.T.shape[1]] = lmpc.slackObs.T
            self.rec.boundary_violation_record.append(obsSlack)

            self.rec.xy_states = np.asarray(self.rec.xy_states)
            values = np.asarray(self.rec.laptimes)
            self.rec.xy_lap_record.append(np.concatenate([np.asarray(self.rec.xy_states), values[:, None]], axis=1))
            self.rec.a_opt_records.append(self.rec.a_opt_record)
            self.rec.laptime_record.append(laptime)
            self.rec.ss_average_record.append(np.mean(self.rec.ss_violation_record))
            self.rec.ss_max_record.append(np.max(self.rec.ss_violation_record))
            self.rec.boundary_average_record.append(np.mean(self.rec.boundary_violation_record))
            self.rec.boundary_max_record.append(np.max(self.rec.boundary_violation_record))
            self.rec.control_records.append(np.asarray(self.rec.control_record))
            self.rec.mean_vels.append(np.mean(np.linalg.norm(self.rec.xy_states[:, 2:4])))
            if laptime < self.rec.min_laptime:
                self.rec.min_laptime = laptime
                self.rec.min_states = self.rec.xy_states

            self.log.line(itCounter, 'mean speed', self.rec.mean_vels[-1], 
                    'laptime', laptime, print_line=self.verbose)
            self.log.line('average ss_violation:', np.mean(self.rec.ss_violation_record), 'max', np.max(self.rec.ss_violation_record),
                'average boundary_violation:', np.mean(self.rec.boundary_violation_record), 'max', np.max(self.rec.boundary_violation_record), print_line=self.verbose)
            
            xcl = np.array(xcl).T
            ucl = np.array(ucl).T
            lmpc.addTrajectory(xcl, ucl)

            # plt.plot(xcl[0], xcl[1], 'r')
            # plt.plot(self.xclFeasible[0], self.xclFeasible[1], 'b')
            # plt.show()

            print(f"Iteration {itCounter} completed")
            itCounter += 1
            # Store time and cost
            meanTimeCostLMPC.append(np.array([np.sum(timeLMPC)/lmpc.cost, lmpc.cost]))
            
            np.savetxt('storedData/closedLoopIteration'+str(itCounter)+'_P_'+str(lmpc.P)+'.txt', np.round(np.array(xcl), decimals=5).T, fmt='%f' )
            np.savetxt('storedData/inputIteration'+str(itCounter)+'_P_'+str(lmpc.P)+'.txt', np.round(np.array(ucl), decimals=5).T, fmt='%f' )
            np.savetxt('storedData/meanTimeLMPC_P_'+str(lmpc.P)+'.txt', np.array(meanTimeCostLMPC), fmt='%f' )
                
            laptime = 0.0
            self.rec.init('laptimes', # record per action laptime,
                        'control_record', 
                        'cross_flags',
                        's_states',
                        'xy_states',
                        'mppi_state_record',
                        'mppi_cov_record',
                        'ss_violation_record',
                        'boundary_violation_record',
                        'lamb_record')

        print('max lap reached')
        plottings.plot_trajectory('', laptime)
        plottings.plot_trajectory_history(itCounter, laptime)
        plottings.plot_laptime_speed('', laptime)
        plottings.plot_ss_violation('', laptime)
        plottings.plot_boundary_violation('', laptime)
        plottings.plot_speed_position('', laptime)
        plottings.plot_speed_position_history(itCounter, laptime)
        plottings.plot_acceleration_position('', laptime) 
        plottings.plot_acceleration_position_history(itCounter, laptime)
        return end_action(self.rec, self.save_dir, plottings)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, xclFeasible):
        super().__init__()
        self.xclFeasible = xclFeasible
        self.initUI()

    def initUI(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        self.window = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.window)
        self.setWindowTitle("F1Tenth Gym")
        self.setGeometry(0, 0, 640, 640)
        self.canvas = self.window.addPlot()

        # Plot initial feasible trajectory
        self.waypoints_plot = self.canvas.plot(
            self.xclFeasible[0, :],
            self.xclFeasible[1, :],
            pen=pg.mkPen(color=(0, 0, 255), width=2),
            name="Initial Feasible Trajectory",
        )

        # Plot obstacle (a circle around [30, 0] with radius 10)
        theta = np.linspace(0, 2 * np.pi, 100)
        x_obs = 30 + 10 * np.cos(theta)
        y_obs = 0 + 10 * np.sin(theta)
        self.obs_plot = self.canvas.plot(
            x_obs,
            y_obs,
            pen=None,
            symbol="o",
            symbolPen=pg.mkPen(color=(0, 0, 0), width=0),
            symbolBrush=pg.mkBrush(color=(0, 0, 0)),
            symbolSize=5,
            name="Obstacle",
        )

        # Plot current position
        self.current_location_plot = self.canvas.plot(
            [0], [0], pen=None,
            symbol="o",
            symbolPen=pg.mkPen(color=(255, 0, 0), width=0),
            symbolBrush=pg.mkBrush(color=(255, 0, 0)),
            symbolSize=5,
            name="Current Position",
        )
        
		# Plot optimal trajectory
        self.optimal_trajectory_plot = self.canvas.plot(
            [0], [0],
			pen=pg.mkPen(color=(0, 255, 0), width=2),
			name="Optimal Trajectory",
		)

        # Add legend
        self.legend = self.canvas.addLegend()
        self.legend.addItem(self.waypoints_plot, "Initial Feasible Trajectory")
        self.legend.addItem(self.obs_plot, "Obstacle")
        self.legend.addItem(self.current_location_plot, "Current Position")

        # Timer to update current position
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000 // 60)

    def update_plot(self):
        if curr_pos:
            self.current_location_plot.setData(np.array(curr_pos))
        if curr_opt_traj is not None:
            self.optimal_trajectory_plot.setData(curr_opt_traj[0, :], curr_opt_traj[1, :])
            


def main():
    if not os.path.exists('storedData'):
        os.makedirs('storedData')

    plot = True
    N = 10
    dt = 0.5
    x0 = [0, 0, 0, 0]
    itMax = 20
    
    ftocp = FTOCP(N=N, dt=dt)
    xclFeasible, uclFeasible = feasTraj(ftocp)
    np.savetxt('storedData/closedLoopFeasible.txt',xclFeasible.T, fmt='%f' )
    np.savetxt('storedData/inputFeasible.txt',uclFeasible.T, fmt='%f' )

    if plot:
        app = QtWidgets.QApplication([])

        def signal_handler(sig, frame):
            print('Exiting...')
            app.quit()

        # Register Ctrl+C handler
        signal.signal(signal.SIGINT, signal_handler)
        
        window = MainWindow(xclFeasible)
        window.show()

        lmpc_thread = LMPCTask(xclFeasible, uclFeasible, ftocp, x0, itMax, N=N, plot=plot, safeSetOption='close')
        lmpc_thread.update_signal.connect(window.update_plot)
        lmpc_thread.start()

        app.exec()
    else:
        LMPCTask(xclFeasible, uclFeasible, ftocp, x0, itMax, N=N, plot=plot, safeSetOption='close').run()


def feasTraj(ftocp):
    A = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    B = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 1]])
    ss = ct.ss(A, B, np.eye(4), np.zeros((4, 2)))
    ss = ct.c2d(ss, 0.5)

    Q = np.diag([0.1, 0.1, 10, 10])
    R = 100 * np.eye(2)
    K, _, _ = ct.lqr(ss, Q, R)

    x0 = np.array([0, 0, 0, 0])
    xcl = [x0]
    ucl = []

    xgoal = np.array([30, 30, 0, 0])
    while np.linalg.norm(xcl[-1][:2] - xgoal[:2]) > 0.2:
        u = -K @ (xcl[-1] - xgoal)
        u = np.clip(u, -1, 1)
        xcl.append(ftocp.f(xcl[-1], u))
        ucl.append(u)

    xgoal = np.array([60, 0, 0, 0])
    while np.linalg.norm(xcl[-1][:2] - xgoal[:2]) > 0.1:
        u = -K @ (xcl[-1] - xgoal)
        u = np.clip(u, -1, 1)
        xcl.append(ftocp.f(xcl[-1], u))
        ucl.append(u)

    return np.array(xcl).T, np.array(ucl).T


if __name__ == "__main__":
    main()
