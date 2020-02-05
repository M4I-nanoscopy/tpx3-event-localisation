import sys
import h5py
import tkinter as Tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)
import matplotlib.pyplot as plt
# from keras.models import load_model

from trajectories import plot_3dtrajectory
from pixels import plot_pixels


# from deeplearning import VisualiseConvLayer

def _quit():
    root.quit()
    root.destroy()


def go():
    global slider, index_val

    slider.set(int(index_val.get()))


def set_plot(i):
    global idx, slider, index_val

    index_val.set(i)
    idx = int(i)
    show_plots(idx)


def prev_plot():
    global idx, slider

    idx = idx - 1
    slider.set(idx)


def next_plot():
    global idx, slider

    idx = idx + 1
    slider.set(idx)


def get_pixel(i):
    global pixels

    if pixels is None:
        return None

    return pixels[i]


def get_incident(i):
    global incidents

    if incidents is None:
        return None

    return incidents[i]


def get_edges(i):
    global edges

    if edges is None:
        return None

    return edges[i]


def get_trajectory(i):
    global trajectories

    if trajectories is None:
        return None

    return trajectories[str(i)][()]


def get_prediction(i):
    global predictions

    if predictions is None:
        return None

    pred = dict()
    for label in predictions:
        pred[label] = predictions[label][i]

    return pred


# def get_actv(model, i, pixel):
#     weights = VisualiseConvLayer.getWeightsLayer(model, 0)
#     actvs = VisualiseConvLayer.get_activations(model, pixel.reshape(1, 2, 10, 10), layer_name='separable_conv2d_1')
#
#     return weights, actvs

def show_plots(i):
    global canvas, canvas_pix, sensor_height

    pixel = get_pixel(i)
    trajectory = get_trajectory(i)
    prediction = get_prediction(i)
    incident = get_incident(i)
    # weights, actvs = get_actv(i, pixel)
    edges = get_edges(i)

    if trajectory is not None:
        traj_fig.clear()
        plot_3dtrajectory.plot(traj_fig, trajectory, sensor_height)
        canvas.draw()

    if pixel is not None:
        # Get pixels
        pix_fig.clear()
        plot_pixels.plot(pix_fig, pixel, incident, prediction, edges)
        canvas_pix.draw()

    # if actvs is not None:
    #     # Get pixels
    #     traj_fig.clear()
    #     VisualiseConvLayer.display_activations(actvs, weights, traj_fig)
    #     canvas.draw()


# File
filename = sys.argv[1]
f = h5py.File(filename, 'r')

pixels, predictions, trajectories, incidents, edges = None, None, None, None, None
if 'clusters' in f:
    pixels = f['clusters'][()]
if 'trajectories' in f:
    trajectories = f['trajectories']
if 'predictions' in f:
    predictions = f['predictions']
if 'incidents' in f:
    incidents = f['incidents'][()]
if 'edges' in f:
    edges = f['edges'][()]

sensor_height = f.attrs['sensor_height'] if 'sensor_height'  in f.attrs else 300000

# Setup Tk
root = Tk.Tk()
root.wm_title("Trajectory Browser")
graphs = Tk.Frame(root)

# Setup info screen

info = Tk.Frame(graphs)
info.grid(row=0, column=0, sticky='N')
Tk.Label(master=info, text="Information:", height=2).pack(side=Tk.TOP)

attrs = f.attrs
Tk.Label(master=info, text="Sensor height: %s" % attrs.get('sensor_height', 'N/A'), anchor='w').pack(side=Tk.TOP)
Tk.Label(master=info, text="Sensor material: %s" % attrs.get('sensor_material', 'N/A'), anchor='w').pack(side=Tk.TOP,
                                                                                                         fill='both')
Tk.Label(master=info, text="Beam energy: %s" % attrs.get('beam_energy', 'N/A'), anchor='w').pack(side=Tk.TOP,
                                                                                                 fill='both')
Tk.Label(master=info, text="Source: %s" % attrs.get('data_source', 'N/A'), anchor='w').pack(side=Tk.TOP, fill='both')

# Setup graphs

# actv_fig = plt.figure()
# canvas_actv = FigureCanvasTkAgg(actv_fig, master=graphs)
# canvas_actv.draw()
# toolbar_frame_actv = Tk.Frame(graphs)
# toolbar = NavigationToolbar2Tk(canvas_actv, toolbar_frame_actv)
# toolbar.update()
# canvas_actv.get_tk_widget().grid(row=0, column=3)
# toolbar_frame_actv.grid(row=1, column=3, sticky=Tk.W)

traj_fig = plt.figure()
canvas = FigureCanvasTkAgg(traj_fig, master=graphs)
canvas.draw()
toolbar_frame_traj = Tk.Frame(graphs)
toolbar = NavigationToolbar2Tk(canvas, toolbar_frame_traj)
toolbar.update()
canvas.get_tk_widget().grid(row=0, column=2)
toolbar_frame_traj.grid(row=1, column=2, sticky=Tk.W)

pix_fig = plt.figure()
canvas_pix = FigureCanvasTkAgg(pix_fig, master=graphs)
canvas_pix.draw()
toolbar_frame_pix = Tk.Frame(graphs)
toolbar = NavigationToolbar2Tk(canvas_pix, toolbar_frame_pix)
toolbar.update()
canvas_pix.get_tk_widget().grid(row=0, column=1)
toolbar_frame_pix.grid(row=1, column=1, sticky=Tk.W)

# Controls
controls = Tk.Frame(root)

index_val = Tk.StringVar()
index = Tk.Entry(master=controls, textvariable=index_val).grid(row=0, column=2, sticky='S')
go = Tk.Button(master=controls, text='Go', command=go).grid(row=0, column=3, sticky='S')
next = Tk.Button(master=controls, text='Prev', command=prev_plot).grid(row=1, column=1, sticky='S')
slider = Tk.Scale(
    master=controls,
    from_=0, to=f['clusters'].shape[0],
    orient=Tk.HORIZONTAL, length=400,
    command=set_plot,
    showvalue=0
)
slider.grid(row=1, column=2)
prev = Tk.Button(master=controls, text='Next', command=next_plot).grid(row=1, column=3, sticky='S')

# Tk final setup
root.protocol("WM_DELETE_WINDOW", _quit)

graphs.pack(side=Tk.TOP)
controls.pack(side=Tk.BOTTOM)

if 2 in sys.argv and int(sys.argv[2]) > 0:
    idx = int(sys.argv[2])
else:
    idx = 0

set_plot(idx)
slider.set(idx)

Tk.mainloop()
