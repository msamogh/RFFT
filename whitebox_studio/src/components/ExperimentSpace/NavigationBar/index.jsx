import React from 'react';
import './NavigationBar.css';

const NavigationBar = (props) => (
  <div className="NavigationBar">
    <button className="NavigationBar-button" onClick={props.onClick('Train')}> Train </button>
    <button className="NavigationBar-button" onClick={props.onClick('Annotate')}> Annotate </button>     
    {/* <button className="NavigationBar-button" onClick={props.onClick('Explain')}> Explain </button>        */}
  </div>
);

export default NavigationBar;
