### Docker Postgres Reference

# Pull the latest PostgreSQL image
docker pull postgres

# Creates a new Docker container named 'postgres_data' using the 'alpine' image.
# This container will serve as a data volume container to persist PostgreSQL data.
docker create -v /var/lib/postgresql/data --name postgres_data alpine

# Runs a new PostgreSQL container named 'postgres_container' in detached mode (-d).
# Sets the environment variable POSTGRES_PASSWORD to 'pwd'.
# Maps port 5432 on the host to port 5432 on the container.
# Uses the 'postgres_data' container to persist data.
docker run -d -e POSTGRES_PASSWORD=pwd -p 5432:5432 --name postgres_container --volumes-from postgres_data postgres

# Opens an interactive terminal session inside the 'postgres_container'
docker exec -it postgres_container bash

# Connects to the PostgreSQL server using the 'psql' command-line tool as the 'postgres' user
psql -U postgres

# Creates a new database named 'ps_testing_db'.
CREATE DATABASE ps_testing_db;

# Executes a SQL command to create a new user with the specified username and password.
# The command is run by the specified admin user on the specified database.
psql -c "CREATE USER {username} WITH PASSWORD '{password}';" -U {admin_username} -d {database_name}

# Lists all databases in the PostgreSQL server.
psql -c "\l" -U postgres

# Connects to the 'mydatabase' database as the 'postgres' user.
psql -U postgres -d mydatabase
