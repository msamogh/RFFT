
import React from 'react';
import Highcharts from 'highcharts';
import {
  HighchartsChart, Chart, withHighcharts, XAxis, YAxis, Title, Legend, LineSeries
} from 'react-jsx-highcharts';

const plotOptions = {
  series: {
    pointStart: 0
  }
};

const chart = {
  height: 300,
  width: 450,
  spacing: [5, 5, 5, 5]
}

const App = () => (
  <div className="app">
    <h4>Loss Function</h4>
    <HighchartsChart plotOptions={plotOptions} chart={chart}>
      <Chart />

      <Title>Loss Function</Title>

      <Legend layout="vertical" align="right" verticalAlign="middle" />

      <XAxis>
        <XAxis.Title>Epochs</XAxis.Title>
      </XAxis>

      <YAxis id="number">
        <YAxis.Title>Error</YAxis.Title>
        <LineSeries id="error" name="error" data={[100, 90, 85, 84, 81, 60, 40, 45, 50, 30, 10, 1, 0.1, 0.1]} />
        <LineSeries id="error_inverse" name="error_inverse" data={[ 0, 10, 15, 16, 19, 40, 60, 55, 50, 70, 90, 99, 99.9, 99.9 ]} />
      </YAxis>
    </HighchartsChart>
  </div>
);

export default withHighcharts(App, Highcharts);