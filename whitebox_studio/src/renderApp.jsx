import React from 'react';
import ReactDOM from 'react-dom';
import App from './components/App';

const renderApp = () => {
  ReactDOM.render(<App />,document.getElementById('root'));
};

export default renderApp;
