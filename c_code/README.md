# Generate c code

I used in this project a reccurent neural network to generate c code based on a dataset of c files from from the <a href="https://github.com/torvalds/linux">linux repository</a>.

More Information about the project could be find on my medium article : <a href=""> Here </a>

<a href="" ><img src="img/leo_vallet.jpeg" /></a>
<center><a href="https://www.linkedin.com/in/leovallet/">Illustration by Léo Vallet</a></center>

## Requirements

<ul>
<li>Tensorflow</li>
<li>Numpy</li>
<li>Docopt</li>
<li>Matplotlib</li>
</ul>

## How to train the model

 
    $> python train.py

This file will create two new folders. The first one is the logs folder, you can use tensorboard on it, the other one is the checkpoints folder where the progression of the model is saved. A new checkpoint is saved at the end of each epoch.

## How to use the model

Once your model is trained, you can try to generate a new c file.

    $> python use.py -h
    $> python use.py <path-to-you-ckpt-file>

## Example of function automatically created with this project


    /*
     * Wake_update_const in a completity the allocation files
     * @copurate: if the fault the polynomial semaphores.
     */
    static ssize_t
    remove_slot(struct kioctx *ctx, const char *buf * int id, const char *buf, int len, int len)
    {
      int inode = sec_free(sector);
      spin_unlock(&c->erase_completion_lock);

      if (!c->resectors[i].size)
        return -EINVAL;

      mutex_unlock(&ctx->ring_lock);
      return err;
    }
