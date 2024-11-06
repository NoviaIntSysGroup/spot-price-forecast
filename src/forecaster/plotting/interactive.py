import plotly.graph_objects as go

from forecaster.data import fingrid


def plot_predictions(spot_data, predictions=[], slider=False):

    # plot the data
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=spot_data.index, y=spot_data.y, mode='lines', name='Spot Price'))
    if predictions:
        fig.add_trace(go.Scattergl(x=predictions['test'].index, y=predictions['test'].values, mode='lines', name='Predictions (Test)'))
        fig.add_trace(go.Scattergl(x=predictions['train'].index, y=predictions['train'].values, mode='lines', name='Predictions (Train)'))
    # add zoom
    if slider:
        fig.update_layout(xaxis_rangeslider_visible=True)
    fig.update_layout(title='Spot Price Over Time', xaxis_title='Time', yaxis_title='Spot Price')
    fig.show()

def plot_external_data(dataset_ids):

    # plot the data
    fig = go.Figure()

    for dataset_id in dataset_ids:
        data = fingrid.load_fingrid_data(dataset_id)
        fig.add_trace(go.Scattergl(x=data.index, y=data.values.flatten(), mode='lines', name=data.columns[0]))

    fig.update_layout(title='External Variables Over Time', xaxis_title='Time', yaxis_title='MW')
    fig.show()