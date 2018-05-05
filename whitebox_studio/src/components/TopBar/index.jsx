import React from 'react';
import './TopBar.css';

const TopBar = (props) => (
  <div className="TopBar">
    <button className="TopBar-home-button" onClick={props.goHome}>WHITE Box</button>
  </div>
);

export default TopBar;
