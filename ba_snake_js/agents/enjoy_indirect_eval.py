from ppo1_enjoy import main


if __name__ == '__main__':
    actor_name = 'Planar-direction-v1-2'
    env_name = 'Planar-direction-enjoy-circle-v0'
    num_enjoys = 10
    main(actor_name=actor_name, env_name=env_name, num_enjoys=num_enjoys)