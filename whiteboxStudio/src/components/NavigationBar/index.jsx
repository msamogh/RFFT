import React from 'react';
import './NavigationBar.css';

const NavigationBarButton = ({title, onClick}) => (
  <button className="NavigationBar-Button" onClick={onClick}> {title} </button>
);

const NavigationBarHeading = ({title}) => (
  <div className="NavigationBar-Heading"> {title} </div>
);

const NavigationBar = ({onClick}) => (
  <div className="NavigationBar">
    <div className="NavigationBar-section">
      <NavigationBarHeading title="White Box Studio"/>
      <NavigationBarButton title="Home" onClick={()=>onClick('home')}/>
    </div>
    <div className="NavigationBar-section">
      <NavigationBarHeading title="Tools"/>
      <NavigationBarButton title="image masking" onClick={()=>onClick('imageMasker')}/>
      <NavigationBarButton title="text anotator" onClick={()=>onClick('textAnotator')}/>  
    </div>    
  </div>
);

export default NavigationBar;
