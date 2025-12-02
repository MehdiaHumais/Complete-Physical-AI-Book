import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function Hero() {
  return (
    <header className={clsx('hero', styles.heroBanner)} style={{minHeight: '80vh', display: 'flex', alignItems: 'center', textAlign: 'center'}}>
      <div className="container">
        
        {/* Badge */}
        <div style={{
          background: 'rgba(249, 115, 22, 0.1)', 
          color: '#f97316', 
          display: 'inline-block', 
          padding: '6px 16px', 
          borderRadius: '50px', 
          fontWeight: 'bold', 
          marginBottom: '1.5rem', 
          border: '1px solid rgba(249, 115, 22, 0.3)',
          textTransform: 'uppercase',
          letterSpacing: '1px'
        }}>
          Panaversity Official
        </div>

        {/* Main Title */}
        <Heading as="h1" style={{fontSize: '5rem', lineHeight: '1.1', marginBottom: '1.5rem', fontFamily: 'Space Grotesk', color: 'white'}}>
          Physical AI <span style={{color: '#f97316'}}>&</span> Robots
        </Heading>

        {/* Subtitle */}
        <p style={{fontSize: '1.5rem', color: '#94a3b8', marginBottom: '3rem', maxWidth: '700px', margin: '0 auto 3rem auto'}}>
          The complete engineering handbook for Physical AI, ROS 2, and Humanoid Robotics.
        </p>

        {/* Buttons (Centered) */}
        <div style={{display: 'flex', gap: '20px', justifyContent: 'center'}}>
          
          {/* Button 1: Start Reading */}
          {/* Note: This points to your FIRST MODULE to ensure it works */}
          {/* <Link 
            className="button button--primary button--lg" 
            to="/docs/Module-1-Foundations/embodied-intelligence"
            style={{padding: '15px 40px', fontSize: '18px'}}>
            Start Reading
          </Link> */}
           <Link className="button button--primary button--lg" to="/docs/intro" style={{padding: '15px 40px', fontSize: '18px'}}>
                Start Reading
              </Link>
          {/* Button 2: GitHub */}
          <Link 
            className="button button--secondary button--lg" 
            to="https://github.com/MehdiaHumais/Complete-Physical-AI-Book"
            style={{padding: '15px 40px', fontSize: '18px', border: '1px solid #475569', color: 'white', background: 'transparent'}}>
            GitHub Repo
          </Link>

        </div>

      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title="Physical AI" description="Robotics Textbook">
      <Hero />
    </Layout>
  );
}